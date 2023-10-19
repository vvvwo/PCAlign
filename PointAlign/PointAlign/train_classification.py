import numpy as np
import importlib
import argparse
from utils.main_utils import *
from tqdm import tqdm
from dataset.ModelNet.ModelNetDataLoader import ModelNetDataset, my_collate_fn
import torch.utils.data


def parse_args():
    parser = argparse.ArgumentParser('cls training')
    parser.add_argument('--model', type=str, default='PointNet/PointNet_Cls', help='model name')
    parser.add_argument('--gpu', type=str, default=[0], help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--input_dim', type=int, default=6, help='input dimension')
    parser.add_argument('--data_path', default='data/ModelNet40', type=str, help='path store data')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampling')
    parser.add_argument('--feature_transform', default=True, help='Whether to transform the feature')
    return parser.parse_args()


def train(classifier, trainDataLoader, optimizer, criterion, epoch, logger, num_class, device_main):
    mean_correct = []
    loss_all = 0
    train_class_acc = np.zeros((num_class, 3))
    classifier = classifier.train()
    for _, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        optimizer.zero_grad()
        points, target = points.float().to(device_main), target.long().to(device_main)
        # --------------------------------------------------------------------------------------------------------------
        # 预测
        # --------------------------------------------------------------------------------------------------------------
        pred = classifier(points.permute(0, 2, 1))
        loss = criterion(pred, target)
        loss_all += loss.item()
        pred_choice = pred.data.max(1)[1][:, None]
        # ----------------------------------------------------------------------------------------------------------
        # 计算两个指标
        # ----------------------------------------------------------------------------------------------------------
        for cat in np.unique(target.cpu()):
            accuracy = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            train_class_acc[cat, 0] += accuracy.item() / float(target[target == cat].size()[0])
            train_class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
        loss.backward()
        optimizer.step()
    loss_mean = loss_all / len(trainDataLoader)
    train_instance_acc = np.mean(mean_correct)
    train_class_acc[:, 2] = train_class_acc[:, 0] / train_class_acc[:, 1]
    train_class_acc[np.isnan(train_class_acc)] = 0
    train_class_acc = np.mean(train_class_acc[:, 2])
    log_string(logger, 'Epoch:{}, Loss:{}, Train Instance Accuracy:{}, Train Class Accuracy{}'.format(epoch, loss_mean,
                                                                                                      train_instance_acc,
                                                                                                      train_class_acc))
    return train_instance_acc, train_class_acc


def test(model, testDataloader, criterion, logger, epoch, num_class, device_main):
    mean_correct = []
    loss_all = 0
    test_class_acc = np.zeros((num_class, 3))
    classifier = model.eval()
    with torch.no_grad():
        for _, (points, target) in tqdm(enumerate(testDataloader), total=len(testDataloader)):
            points, target = points.float().to(device_main), target.long().to(device_main)
            # ----------------------------------------------------------------------------------------------------------
            # 预测
            # ----------------------------------------------------------------------------------------------------------
            pred = classifier(points.permute(0, 2, 1))
            loss = criterion(pred, target)
            loss_all += loss.item()
            pred_choice = pred.data.max(1)[1][:, None]
            # ----------------------------------------------------------------------------------------------------------
            # 计算两个指标
            # ----------------------------------------------------------------------------------------------------------
            for cat in np.unique(target.cpu()):
                accuracy = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                test_class_acc[cat, 0] += accuracy.item() / float(target[target == cat].size()[0])
                test_class_acc[cat, 1] += 1
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
        # ----------------------------------------------------------------------------------------------------------
        # 输出结果
        # ----------------------------------------------------------------------------------------------------------
        loss_mean = loss_all / len(testDataloader)
        test_instance_acc = np.mean(mean_correct)
        test_class_acc[:, 2] = test_class_acc[:, 0] / test_class_acc[:, 1]
        test_class_acc[np.isnan(test_class_acc)] = 0
        test_class_acc = np.mean(test_class_acc[:, 2])
        log_string(logger,
                   'Epoch:{}, Loss:{}, Test Instance Accuracy:{}, Test Class Accuracy{}'.format(epoch, loss_mean,
                                                                                                test_instance_acc,
                                                                                                test_class_acc))
    return test_instance_acc, test_class_acc


def main(args):
    # ------------------------------------------------------------------------------------------------------------------
    # 超参数
    # ------------------------------------------------------------------------------------------------------------------
    device_main = torch.device('cuda:{}'.format(args.gpu[0]))
    # ------------------------------------------------------------------------------------------------------------------
    # 创建目录
    # ------------------------------------------------------------------------------------------------------------------
    log_dir, exp_dir, checkpoints_dir = create_dir(args, "cls")
    # ------------------------------------------------------------------------------------------------------------------
    # 创建输出日志
    # ------------------------------------------------------------------------------------------------------------------
    logger = create_logger(args, log_dir)
    log_string(logger, 'PARAMETER ...')
    log_string(logger, args)
    # ------------------------------------------------------------------------------------------------------------------
    # 加载数据
    # ------------------------------------------------------------------------------------------------------------------
    log_string(logger, 'Load dataset ...')
    train_dataset = ModelNetDataset(args=args, split='train')
    test_dataset = ModelNetDataset(args=args, split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=0, drop_last=True, pin_memory=True,
                                                  collate_fn=lambda x: my_collate_fn(x, train=True))
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0, drop_last=True, pin_memory=True,
                                                 collate_fn=lambda x: my_collate_fn(x, train=False))
    # ------------------------------------------------------------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------------------------------------------------------------
    model = importlib.import_module('models.{}.{}'.format(args.model.split("/")[0], args.model.split("/")[1]))
    classifier = model.PointNetCls(args).to(device_main)
    criterion = model.get_loss().to(device_main)
    # ------------------------------------------------------------------------------------------------------------------
    # 加载权重
    # ------------------------------------------------------------------------------------------------------------------
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string(logger, 'Use pretrain model...')
    except FileNotFoundError:
        log_string(logger, 'No existing model, starting training from scratch...')
        start_epoch = 0
    # ------------------------------------------------------------------------------------------------------------------
    # 优化器+学习率
    # ------------------------------------------------------------------------------------------------------------------
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(),
                                     lr=args.learning_rate,
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=0.0001)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=args.decay_rate)
    # ------------------------------------------------------------------------------------------------------------------
    # 训练+测试
    # ------------------------------------------------------------------------------------------------------------------
    best_epoch = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        train_instance_acc, train_class_acc = train(classifier, trainDataLoader, optimizer, criterion, epoch, logger,
                                                    args.num_category, device_main)
        scheduler.step()
        test_instance_acc, test_class_acc = test(classifier, testDataLoader, criterion, logger, epoch,
                                                 args.num_category, device_main)
        # --------------------------------------------------------------------------------------------------------------
        # 更新精度
        # --------------------------------------------------------------------------------------------------------------
        if test_instance_acc >= best_instance_acc:
            best_instance_acc = test_instance_acc
            best_epoch = epoch + 1
        if test_class_acc >= best_class_acc:
            best_class_acc = test_class_acc
        log_string(logger,
                   'Epoch:{}, Best Instance Accuracy:{}, Best Class Accuracy:{}'.format(epoch, best_instance_acc,
                                                                                        best_class_acc))
        # --------------------------------------------------------------------------------------------------------------
        # 保存模型
        # --------------------------------------------------------------------------------------------------------------
        if test_instance_acc >= best_instance_acc:
            log_string(logger, 'Save model...')
            save_path = str(checkpoints_dir) + '/best_model.pth'
            log_string(logger, 'Saving at %s' % save_path)
            state = {'epoch': best_epoch,
                     'train_instance_acc': train_instance_acc,
                     'train_class_acc': train_class_acc,
                     'best_instance_acc': best_instance_acc,
                     'best_class_acc': best_class_acc,
                     'model_state_dict': classifier.state_dict()}
            torch.save(state, save_path)
        log_string(logger, "------------------------------------------------------------------------------------------")
    logger.info('End of training...')


if __name__ == '__main__':
    main(parse_args())

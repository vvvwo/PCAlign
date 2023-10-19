import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # --------------------------------------------------------------------------------------------------------------
        # 通过添加一个单位矩阵，可以在训练初期提供一个合适的起点，使网络更容易学习适应输入数据的旋转变换。可以提高网络训练的稳定性和收敛速度。
        # --------------------------------------------------------------------------------------------------------------
        identity = torch.eye(3).view(1, 9).repeat(batch_size, 1).float()
        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(self.k).view(1, self.k * self.k).repeat(batch_size, 1).float()
        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, semseg=False):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d() if not semseg else STNkd(k=9)
        self.conv1 = torch.nn.Conv1d(6, 64, 1) if not semseg else torch.nn.Conv1d(9, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.feature_stn = STNkd(k=64)

    def forward(self, x):
        # --------------------------------------------------------------------------------------------------------------
        # 先计算一个通道注意力， (2, 3, 3)
        # --------------------------------------------------------------------------------------------------------------
        global fea_trans
        xyz = x[:, :3, :].transpose(2, 1)
        remain_feature = x[:, 3:, :]
        trans = self.stn(x)
        new_xyz = torch.bmm(xyz, trans).transpose(2, 1)
        x = torch.cat((new_xyz, remain_feature), dim=1)
        # --------------------------------------------------------------------------------------------------------------
        # 对输入的点云坐标数据提取特征
        # (2, 6, 1024) -> (2, 64, 1024)
        # --------------------------------------------------------------------------------------------------------------
        x = F.relu(self.bn1(self.conv1(x)))
        # --------------------------------------------------------------------------------------------------------------
        # 再学习一个通道注意力，(2, 64, 64)
        # --------------------------------------------------------------------------------------------------------------
        if self.feature_transform:
            fea_trans = self.feature_stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, fea_trans)
            x = x.transpose(2, 1)
        # --------------------------------------------------------------------------------------------------------------
        # (2, 64, 1024) -> (2, 128, 1024) -> (2, 1024, 1024) -> (2, 1024, 1) -> (2, 1024)
        # --------------------------------------------------------------------------------------------------------------
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x


class PointNetCls(nn.Module):
    def __init__(self, args):
        super(PointNetCls, self).__init__()
        self.num_category = args.num_category
        self.feature_transform = args.feature_transform
        self.feat = PointNetEncoder(global_feat=True, feature_transform=self.feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_category)
        self.dropout = nn.Dropout(p=0.3)
        self.dp1 = nn.Dropout(p=0.3)
        self.dp2 = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # --------------------------------------------------------------------------------------------------------------
        # global_feat: (2, 1024)
        # --------------------------------------------------------------------------------------------------------------
        x = self.feat(x)
        # --------------------------------------------------------------------------------------------------------------
        # 分类头(2, 1024) -> (2, 40)
        # --------------------------------------------------------------------------------------------------------------
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x).reshape(-1, 4, self.num_category)
        # --------------------------------------------------------------------------------------------------------------
        # 选择最佳点云
        # --------------------------------------------------------------------------------------------------------------
        out = select_matching_point_cls(x)
        return out


def select_matching_point_cls(logit):
    """
    从四个点云副本中选择最佳点云副本--分类任务
    :param logit:
    :return:
    """
    batch_size, num_copies, num_channel = logit.shape[0], logit.shape[1], logit.shape[2]
    all_max_indices = torch.zeros(size=(batch_size, num_copies)).to(logit.device)
    one_hot_value = torch.zeros(size=(batch_size, num_copies, num_channel)).to(logit.device)
    end_value = torch.zeros(size=(batch_size, num_channel)).to(logit.device)
    for b in range(batch_size):
        point_copies = logit[b]
        for i in range(num_copies):
            single_object_copies = point_copies[i, :]
            # ----------------------------------------------------------------------------------------------------------
            # 将每个子列表中的最大值设为1，其他为0
            # ----------------------------------------------------------------------------------------------------------
            max_indices = torch.argmax(single_object_copies, dim=0)
            all_max_indices[b, i] = max_indices
            one_hot_value[b, i, max_indices] = 1
    standard_index = one_hot_value.sum(dim=1).argmax(dim=1)
    standard_index = standard_index.reshape(-1, 1).expand(-1, 4)
    value_choice = (all_max_indices == standard_index).float()
    value_choice = torch.argmax(value_choice, dim=1)
    for b in range(batch_size):
        end_value[b, ] = logit[b, value_choice[b].item(), :]
    return end_value


def select_matching_point_seg(logit):
    """
    从四个点云副本中选择最佳点云副本--分割任务，计算比较快
    @param logit: [2, 4, 2048, 50]
    @return: [2, 2048, 50]
    """
    # ------------------------------------------------------------------------------------------------------------------
    # 选择最佳点云
    # ------------------------------------------------------------------------------------------------------------------
    B, number_points, C = logit.shape[0], logit.shape[1], logit.shape[2]
    end_value = torch.zeros(size=(B, number_points, C)).to(logit.device)
    for b in range(B):
        point_copies = logit[b]
        max_indices = torch.argmax(point_copies, dim=-1)
        result = torch.zeros_like(point_copies)
        result[torch.arange(4).unsqueeze(1), torch.arange(2048).unsqueeze(0), max_indices] = 1
        standard_index = result.sum(dim=0).argmax(dim=1)
        value_choice = max_indices.t() == standard_index.reshape(-1, 1)
        value_choice = torch.argmax(value_choice.float(), dim=1)
        value = point_copies[value_choice, torch.arange(len(value_choice)),]
        end_value[b] = value
    return end_value


def select_matching_point_seg_(logit):
    """
    从四个点云副本中选择最佳点云副本--分割任务，计算比较慢，易于理解
    @param logit: [2, 4, 2048, 50]
    @return: [2, 2048, 50]
    """
    # ------------------------------------------------------------------------------------------------------------------
    # 选择最佳点云
    # ------------------------------------------------------------------------------------------------------------------
    B, number_points, C = logit.shape[0], logit.shape[2], logit.shape[3]
    end_value = torch.zeros(size=(B, number_points, C)).to(logit.device)
    for b in range(B):
        point_copies = logit[b]
        for n in range(number_points):
            copies = point_copies[:, n, :]
            # ----------------------------------------------------------------------------------------------------------
            # 将每个子列表中的最大值设为1，其他为0
            # ----------------------------------------------------------------------------------------------------------
            max_indices = torch.argmax(copies, dim=1)
            result = torch.zeros_like(copies)
            for i, idx in enumerate(max_indices):
                result[i, idx] = 1
            standard_index = result.sum(dim=0).argmax()
            value_choice = torch.nonzero(standard_index == max_indices).min()
            value = copies[value_choice]
            end_value[b, n] = value
    return end_value


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.cls_loss = torch.nn.CrossEntropyLoss()

    def forward(self, pred, label):
        label = label.squeeze()
        cls_loss = self.cls_loss(pred, label)
        # --------------------------------------------------------------------------------------------------------------
        # 计算总损失
        # --------------------------------------------------------------------------------------------------------------
        lambda1 = 1
        total_loss = lambda1 * cls_loss
        return total_loss

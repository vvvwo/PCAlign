import numpy as np


def svd_decomposition(point_cloud):
    """
    # 计算点云的协方差矩阵, 使用SVD对协方差矩阵进行分解，得到特征值和特征向量
    :param point_cloud: N x 3的点云数组，每一行是一个点的XYZ坐标
    :return eigenvalues, eigenvectors: 点云的特征值和特征向量
    """
    # ------------------------------------------------------------------------------------------------------------------
    # 将点云中心化
    # ------------------------------------------------------------------------------------------------------------------
    centered_point_cloud = point_cloud - np.mean(point_cloud, axis=0)
    covariance_matrix = np.dot(centered_point_cloud.T, centered_point_cloud) / (point_cloud.shape[0] - 1)
    # ------------------------------------------------------------------------------------------------------------------
    # 函数返回的特征值和特征向量默认是升序排列的，我们反转它们以获得降序排列
    # ------------------------------------------------------------------------------------------------------------------
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    return eigenvalues, eigenvectors


def transform_coordinate_system(point_cloud, a1, a3):
    """
    # 构建坐标系, 将点云进行转换
    :param point_cloud: N x 3的点云数组，每一行是一个点的XYZ坐标
    :param a1: 特征值最大对应的特征向量
    :param a3: 特征值最小对应的特征向量
    :return transformed_point_cloud: N x 3的点云数组，新的坐标系下的点云XYZ坐标
    """
    # ------------------------------------------------------------------------------------------------------------------
    # 计算第三个轴，a2 = a1 × a3
    # ------------------------------------------------------------------------------------------------------------------
    a2 = np.cross(a1, a3)
    # ------------------------------------------------------------------------------------------------------------------
    # 构建旋转矩阵，将点云投影到新的坐标系
    # ------------------------------------------------------------------------------------------------------------------
    rotation_matrix = np.vstack((a1, a2, a3)).T
    transformed_point_cloud = np.dot(point_cloud, rotation_matrix)
    return transformed_point_cloud


def PCAlign(point_clouds_data):
    """
    生成四个点云对齐后的副本，同时对法向量进行旋转
    :param point_clouds_data: ndarray: (8, 1024, 6)，点云的xyz坐标 + 法向量
    :return transformed_point_clouds: (8, 4, 1024, 3)，经过对齐的点云副本
    """
    point_clouds_xyz = point_clouds_data[:, :, :3]
    point_cloud_normals = point_clouds_data[:, :, 3:]
    num_point_clouds = point_clouds_xyz.shape[0]
    transformed_point_clouds = np.zeros((num_point_clouds, 4, point_clouds_data.shape[1], point_clouds_data.shape[-1]))
    for i in range(num_point_clouds):
        # --------------------------------------------------------------------------------------------------------------
        # 获取当前处理的点云和法向量
        # --------------------------------------------------------------------------------------------------------------
        point_cloud_xyz = point_clouds_xyz[i]
        point_cloud_normal = point_cloud_normals[i]
        # --------------------------------------------------------------------------------------------------------------
        # 计算点云的特征值以及特征向量
        # --------------------------------------------------------------------------------------------------------------
        eigenvalues, eigenvectors = svd_decomposition(point_cloud_xyz)
        # --------------------------------------------------------------------------------------------------------------
        # 进行点云转换
        # --------------------------------------------------------------------------------------------------------------
        for j in range(4):
            e1_direction = eigenvectors[:, 0] if j % 2 == 0 else -eigenvectors[:, 0]
            e3_direction = (-1) ** (j // 2) * eigenvectors[:, 2]
            # ----------------------------------------------------------------------------------------------------------
            # 对点云进行转换
            # ----------------------------------------------------------------------------------------------------------
            transformed_point_cloud = transform_coordinate_system(point_cloud_xyz, e1_direction, e3_direction)
            transformed_point_clouds[i, j, :, :3] = transformed_point_cloud
            # ----------------------------------------------------------------------------------------------------------
            # 对法向量进行转换
            # ----------------------------------------------------------------------------------------------------------
            transformed_point_normal = transform_coordinate_system(point_cloud_normal, e1_direction, e3_direction)
            transformed_point_clouds[i, j, :, 3:] = transformed_point_normal
    return transformed_point_clouds

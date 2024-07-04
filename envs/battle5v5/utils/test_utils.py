import numpy as np


class Trans:
    @staticmethod
    def rotation_matrix_x(roll):
        """生成绕X轴的旋转矩阵"""
        cos_r = np.cos(roll)
        sin_r = np.sin(roll)
        return np.array([
            [1, 0, 0],
            [0, cos_r, -sin_r],
            [0, sin_r, cos_r]
        ])

    @staticmethod
    def rotation_matrix_y(pitch):
        """生成绕Y轴的旋转矩阵"""
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)
        return np.array([
            [cos_p, 0, sin_p],
            [0, 1, 0],
            [-sin_p, 0, cos_p]
        ])
    @staticmethod
    def rotation_matrix_z(yaw):
        """生成绕Z轴的旋转矩阵"""
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        return np.array([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1]
        ])

    @staticmethod
    def rotation_matrix(pitch, yaw):
        """将机体系向量转换到东北天坐标系"""
        # R_x = Trans.rotation_matrix_x(roll)
        R_y = Trans.rotation_matrix_y(pitch)
        R_z = Trans.rotation_matrix_z(yaw)

        # 组合旋转矩阵
        R = np.dot(R_z, R_y)

        return R

    """输入当前姿态，目标到自身的坐标差，返回目标在自身机体系下向量"""

# distance = 10000
distance = 10000

class CalNineDir:
    @staticmethod
    def get_all_nine_dir(pitch, yaw, cur_x, cur_y, cur_z):
        rotation_matrix = Trans.rotation_matrix(pitch, yaw)
        ans = {}
        for k, one_dir in all_nine_body_dir.items():
            body_dir = np.array(one_dir)
            ned_vector = rotation_matrix @ body_dir
            # 限制向量长度为distance
            length = np.linalg.norm(ned_vector)
            if length > distance or distance > length:
                ned_vector = (ned_vector / length) * distance
            ans[k] = {'X': cur_x + ned_vector[1], 'Y': cur_y + ned_vector[0], 'Z': cur_z - ned_vector[2]}
        return ans

    @staticmethod
    def get_tar_dir(pitch, roll, yaw, dir_x, dir_y, dir_z, cur_x, cur_y, cur_z):
        rotation_matrix = Trans.rotation_matrix(yaw, pitch, roll)
        body_dir = np.array([dir_x, dir_y, dir_z])
        ned_vector = rotation_matrix @ body_dir
        # 限制向量长度为distance
        length = np.linalg.norm(ned_vector)
        if length > 5000 or 5000 > length:
            ned_vector = (ned_vector / length) * 5000
        return {'X': cur_x + ned_vector[1], 'Y': cur_y + ned_vector[0], 'Z': cur_z - ned_vector[2]}


all_nine_body_dir = {
    '0': [1, 0, 0],  # 向前
    '1': [0.5, -1, -0.3],  # 左上
    '2': [0.5, 1, -0.3],  # 右上
    '3': [1, -1, 0.3],  # 左下
    '4': [1, 1, 0.3],  # 右下
    '5': [1, 0, -0.3],  # 向上
    '6': [1, 0, 0.3],  # 向下
    '7': [0, -0.3, 0],  # 向左
    '8': [0, 0.3, 0],  # 向右
}
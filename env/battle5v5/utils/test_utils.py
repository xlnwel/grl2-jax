import math
import numpy as np
import numpy as np


class Trans:

    @staticmethod
    # 机体系转东北天的矩阵
    def rotation_matrix(yaw, pitch, roll):
        # 计算各个旋转矩阵
        R_z = np.array([
            [np.cos(yaw), np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R_y = np.array([
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)]
        ])

        # 组合旋转矩阵
        R = R_z @ R_y @ R_x
        return R

    """输入当前姿态，目标到自身的坐标差，返回目标在自身机体系下向量"""

    @staticmethod
    def test(pitch, roll, yaw, dir_x, dir_y, dir_z):
        # print("pitch:", np.degrees(pitch)," roll:", np.degrees(roll)," yaw:", np.degrees(yaw))
        rotation_matrix = Trans.rotation_matrix(pitch, roll, yaw).T
        tar_self_dir = np.array([dir_y, dir_x, -dir_z])
        body_dir = rotation_matrix @ tar_self_dir

        return {'X': body_dir[0], 'Y': body_dir[1], 'Z': body_dir[2]}


distance = 400


class CalNineDir:
    @staticmethod
    def get_all_nine_dir(pitch, roll, yaw, cur_x, cur_y, cur_z):
        rotation_matrix = Trans.rotation_matrix(pitch, roll, yaw)
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
        rotation_matrix = Trans.rotation_matrix(pitch, roll, yaw)
        body_dir = np.array([dir_x, dir_y, dir_z])
        ned_vector = rotation_matrix @ body_dir
        # 限制向量长度为distance
        length = np.linalg.norm(ned_vector)
        if length > 5000 or 5000 > length:
            ned_vector = (ned_vector / length) * 5000
        return {'X': cur_x + ned_vector[1], 'Y': cur_y + ned_vector[0], 'Z': cur_z - ned_vector[2]}


all_nine_body_dir = {
    '0': [1, 0, 0],  # 向前
    '1': [0.5, -1, -1],  # 左上
    '2': [0.5, 1, -1],  # 右上
    '3': [0.5, -1, 1],  # 左下
    '4': [0.5, 1, 1],  # 右下
    '5': [0.5, 0, -1],  # 向上
    '6': [0.5, 0, 1],  # 向下
    '7': [0.5, -1, 0],  # 向左
    '8': [0.5, 1, 0],  # 向右
}

import math
import numpy as np


class Trans:
    @staticmethod
    def llh_to_ecef(lat, lon, alt) -> {}:
        # WGS84椭球参数
        a = 6378137.0  # 地球半径
        e = 8.1819190842622e-2  # 偏心率

        # 将度转换为弧度
        # lat = math.radians(lat)
        # lon = math.radians(lon)

        N = a / math.sqrt(1 - e ** 2 * math.sin(lat) ** 2)

        X = (N + alt) * math.cos(lat) * math.cos(lon)
        Y = (N + alt) * math.cos(lat) * math.sin(lon)
        Z = ((1 - e ** 2) * N + alt) * math.sin(lat)

        return {'X': X, 'Y': Y, 'Z': Z}

    @staticmethod
    def body_to_ecef(yaw, pitch, roll, vector_body) -> {}:
        # 将角度从度转换为弧度
        # yaw = np.radians(yaw)
        # pitch = np.radians(pitch)
        # roll = np.radians(roll)
        vector_body = np.array([vector_body['X'], vector_body['Y'], vector_body['Z']])
        # 偏航旋转矩阵
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 俯仰旋转矩阵
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # 翻滚旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # 组合旋转
        R = np.dot(R_z, np.dot(R_y, R_x))

        # 将机体系向量转换为ECEF系
        vector_ecef = np.dot(R, vector_body)

        return {'X': vector_ecef[0], 'Y': vector_ecef[1], 'Z': vector_ecef[2]}

    @staticmethod
    def ecef_to_body(yaw, pitch, roll, vector_ecef) -> {}:
        # 将角度从度转换为弧度
        # yaw = np.radians(yaw)
        # pitch = np.radians(pitch)
        # roll = np.radians(roll)
        vector_ecef = np.array([vector_ecef['X'], vector_ecef['Y'], vector_ecef['Z']])
        # 偏航旋转矩阵（逆）
        R_z_inv = np.array([
            [np.cos(yaw), np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 俯仰旋转矩阵（逆）
        R_y_inv = np.array([
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)]
        ])

        # 翻滚旋转矩阵（逆）
        R_x_inv = np.array([
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)]
        ])

        # 组合旋转（注意旋转顺序相反）
        R_inv = np.dot(R_x_inv, np.dot(R_y_inv, R_z_inv))

        # 将ECEF系向量转换为机体系
        vector_body = np.dot(R_inv, vector_ecef)

        return {'X': vector_body[0], 'Y': vector_body[1], 'Z': vector_body[2]}

    @staticmethod
    def ecef_to_lla(x, y, z):
        # WGS-84椭球体常数
        a = 6378137  # 赤道半径
        e = 8.1819190842622e-2  # 第一偏心率

        # 计算经度
        lon = math.atan2(y, x)

        # 迭代计算纬度和高度
        p = math.sqrt(x ** 2 + y ** 2)
        lat = math.atan2(z, p * (1 - e ** 2))
        N = a / math.sqrt(1 - e ** 2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - N

        # 转换为度
        lat = math.degrees(lat)
        lon = math.degrees(lon)

        return lat, lon, alt

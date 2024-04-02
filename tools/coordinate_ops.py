import math

import numpy as np

R = (6378137.0 + 6356752.31424518) / 2.0
# 椭球长半轴长度
m_sdbSemiMajor = 6378137.0
# 椭球短半轴长度
m_sdbSemiMinor = 6356752.3142452157096521378861561
# 椭球第一偏心率
m_sdbe2 = 0.00669437999013
# 椭球第二偏心率
m_sdbep2 = 0.006739496742265


class Coordinate:
    @staticmethod
    def xyz_to_enu(x, y, z):
        return -z, x, y

    @staticmethod
    def getENU(lat, lon, alt):
        """
        :param lat: latitude, in rad
        :param lon: longitude, in rad
        :param alt: altitude, in meter
        :return:
        """
        x, y, z = Coordinate.getXYZ(lat, lon, alt)
        return Coordinate.xyz_to_enu(x, y, z)

    @staticmethod
    def ENU2banwen(e, n, u, heading_rad):
        cos_heading = math.cos(heading_rad)
        sin_heading = math.sin(heading_rad)
        n_ = n * cos_heading + e * sin_heading
        u_ = u
        e_ = -n * sin_heading + e * cos_heading
        return e_, n_, u_

    @staticmethod
    def getXYZ(lat, lon, alt):
        """

        :param lat: latitude, in rad
        :param lon: longitude, in rad
        :param alt: altitude, in meter
        :return:
        """
        # X: n, bei, north,
        # Y: u, tian, sky,
        # Z: -e, dong, east,
        v0 = {'Latitude': 42, 'Longitude': 123, 'Altitude': 0}
        v1 = {'Latitude': lat / math.pi * 180, 'Longitude': lon / math.pi * 180, 'Altitude': alt}
        X, Y, Z = Coordinate.convertLBHToNUE(v0, v1)
        Y = alt
        return X, Y, Z

    @staticmethod
    def convertLBHToNUE(v0, v1):

        DTOR = math.pi / 180

        # 将原点经纬高转为地心坐标
        v0_x, v0_y, v0_z = Coordinate.convertLBHToXYZ(v0['Latitude'] * DTOR, v0['Longitude'] * DTOR, v0['Altitude'])

        # 将飞机当前经纬高转为地心坐标
        vin_x, vin_y, vin_z = Coordinate.convertLBHToXYZ(v1['Latitude'] * DTOR, v1['Longitude'] * DTOR, v1['Altitude'])

        # 将地心坐标转为北东地
        vout_x, vout_y, vout_z = Coordinate.convertXYZToNED(v0_x, v0_y, v0_z, vin_x, vin_y, vin_z)
        # vout_z = v1['Altitude'] - v0['Altitude']

        return vout_x, -vout_z, vout_y

    @staticmethod
    def convertLBHToXYZ(latitude, longitude, altitude):
        sin_latitude = math.sin(latitude)
        cos_latitude = math.cos(latitude)
        sin_longitude = math.sin(longitude)
        cos_longitude = math.cos(longitude)

        N = m_sdbSemiMajor / math.sqrt(1.0 - m_sdbe2 * sin_latitude * sin_latitude)

        X = (N + altitude) * cos_latitude * cos_longitude
        Y = (N + altitude) * cos_latitude * sin_longitude
        Z = (N * (1 - m_sdbe2) + altitude) * sin_latitude

        return X, Y, Z

    @staticmethod
    def convertXYZToNED(x0, y0, z0, x1, y1, z1):
        m_mtx_ned2earth = Coordinate.getMatrixNed2Earth(x0, y0, z0)
        # m_mtx_earth2ned = np.linalg.pinv(m_mtx_ned2earth)
        m_mtx_earth2ned = Coordinate.invers(m_mtx_ned2earth)

        d = 1.0 / (m_mtx_earth2ned[0][3] * x1 + m_mtx_earth2ned[1][3] * y1 + m_mtx_earth2ned[2][3] * z1 +
                   m_mtx_earth2ned[3][3])
        out_x = (m_mtx_earth2ned[0][0] * x1 + m_mtx_earth2ned[1][0] * y1 + m_mtx_earth2ned[2][0] * z1 +
                 m_mtx_earth2ned[3][0]) * d
        out_y = (m_mtx_earth2ned[0][1] * x1 + m_mtx_earth2ned[1][1] * y1 + m_mtx_earth2ned[2][1] * z1 +
                 m_mtx_earth2ned[3][1]) * d
        out_z = (m_mtx_earth2ned[0][2] * x1 + m_mtx_earth2ned[1][2] * y1 + m_mtx_earth2ned[2][2] * z1 +
                 m_mtx_earth2ned[3][2]) * d

        return out_x, out_y, out_z

    @staticmethod
    def getMatrixNed2Earth(x0, y0, z0):
        theta = math.atan2(y0, x0)
        length = math.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)

        if length < 1E-15:
            kethe = 0.0
        else:
            kethe = math.asin(z0 / length) + math.pi / 2

        a = Coordinate.makeRotate(kethe, 0, -1, 0)
        b = Coordinate.makeRotate(theta, 0, 0, 1)
        c = Coordinate.makeTranslate(x0, y0, z0)

        m_mtx_ned2earth = np.matmul(a, b)
        m_mtx_ned2earth = np.matmul(m_mtx_ned2earth, c)

        return m_mtx_ned2earth

    @staticmethod
    def makeRotate(angle, x0, y0, z0):

        m = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        epsilon = 0.0000001

        length = math.sqrt(x0 * x0 + y0 * y0 + z0 * z0)
        if length < epsilon:
            print('-----------error------------')
            return 0, 0, 0, 0

        inversenorm = 1.0 / length
        coshalfangle = math.cos(0.5 * angle)
        sinhalfangle = math.sin(0.5 * angle)

        v = [0, 0, 0, 0]
        v[0] = x0 * sinhalfangle * inversenorm
        v[1] = y0 * sinhalfangle * inversenorm
        v[2] = z0 * sinhalfangle * inversenorm
        v[3] = coshalfangle

        length2 = v[0] ** 2 + v[1] ** 2 + v[2] ** 2 + v[3] ** 2
        if math.fabs(length2) <= 1E-15:
            m[0][0] = 0.0
            m[1][0] = 0.0
            m[2][0] = 0.0
            m[0][1] = 0.0
            m[1][1] = 0.0
            m[2][1] = 0.0
            m[0][2] = 0.0
            m[1][2] = 0.0
            m[2][2] = 0.0
        else:
            if length2 != 1.0:
                rlength2 = 2.0 / length2
            else:
                rlength2 = 2.0

            x2 = rlength2 * v[0]
            y2 = rlength2 * v[1]
            z2 = rlength2 * v[2]

            xx = v[0] * x2
            xy = v[0] * y2
            xz = v[0] * z2

            yy = v[1] * y2
            yz = v[1] * z2
            zz = v[2] * z2

            wx = v[3] * x2
            wy = v[3] * y2
            wz = v[3] * z2

            m[0][0] = 1.0 - (yy + zz)
            m[1][0] = xy - wz
            m[2][0] = xz + wy

            m[0][1] = xy + wz
            m[1][1] = 1.0 - (xx + zz)
            m[2][1] = yz - wx

            m[0][2] = xz - wy
            m[1][2] = yz + wx
            m[2][2] = 1.0 - (xx + yy)

        return m

    @staticmethod
    def makeTranslate(x0, y0, z0):
        return [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [x0, y0, z0, 1]]

    @staticmethod
    def invers(matrix):

        if matrix[0][3] == 0.0 and matrix[1][3] == 0.0 and matrix[2][3] == 0.0 and matrix[3][3] == 1.0:
            return Coordinate.invert_4x3(matrix)
        else:
            return Coordinate.invert_4x4(matrix)

    @staticmethod
    def invert_4x3(mat):
        r00 = mat[0][0]
        r01 = mat[0][1]
        r02 = mat[0][2]

        r10 = mat[1][0]
        r11 = mat[1][1]
        r12 = mat[1][2]

        r20 = mat[2][0]
        r21 = mat[2][1]
        r22 = mat[2][2]

        _mat = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        _mat[0][0] = r11 * r22 - r12 * r21
        _mat[0][1] = r02 * r21 - r01 * r22
        _mat[0][2] = r01 * r12 - r02 * r11

        one_over_det = 1.0 / (r00 * _mat[0][0] + r10 * _mat[0][1] + r20 * _mat[0][2])
        r00 *= one_over_det
        r10 *= one_over_det
        r20 *= one_over_det

        _mat[0][0] *= one_over_det
        _mat[0][1] *= one_over_det
        _mat[0][2] *= one_over_det
        _mat[0][3] = 0.0
        _mat[1][0] = r12 * r20 - r10 * r22  # Have already been divided by det
        _mat[1][1] = r00 * r22 - r02 * r20  # same
        _mat[1][2] = r02 * r10 - r00 * r12  # same
        _mat[1][3] = 0.0
        _mat[2][0] = r10 * r21 - r11 * r20  # Have already been divided by det
        _mat[2][1] = r01 * r20 - r00 * r21  # same
        _mat[2][2] = r00 * r11 - r01 * r10  # same
        _mat[2][3] = 0.0
        _mat[3][3] = 1.0

        r22 = mat[3][3]

        if math.pow((r22 - 1.0), 2) > 1.0e-6:  # Involves perspective, so we must
            # compute the full inverse
            TPinv = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            _mat[3][0] = 0.0
            _mat[3][1] = 0.0
            _mat[3][2] = 0.0

            # define px r00
            # define py r01
            # define pz r02
            # define one_over_s  one_over_det
            # define a  r10
            # define b  r11
            # define c  r12

            a = mat[0][3]
            b = mat[1][3]
            c = mat[2][3]
            r00 = _mat[0][0] * a + _mat[0][1] * b + _mat[0][2] * c
            r01 = _mat[1][0] * a + _mat[1][1] * b + _mat[1][2] * c
            r02 = _mat[2][0] * a + _mat[2][1] * b + _mat[2][2] * c

            # undef a
            # undef b
            # undef c
            # define tx r10
            # define ty r11
            # define tz r12

            r10 = mat[3][0]
            r11 = mat[3][1]
            r12 = mat[3][2]
            one_over_s = 1.0 / (r22 - (r10 * r00 + r11 * r01 + r12 * r02))

            r10 *= one_over_s
            r11 *= one_over_s
            r12 *= one_over_s  # Reduces number of calculations later on

            # Compute inverse of trans * corr
            TPinv[0][0] = r10 * r00 + 1.0
            TPinv[0][1] = r11 * r00
            TPinv[0][2] = r12 * r00
            TPinv[0][3] = -r00 * one_over_s
            TPinv[1][0] = r10 * r01
            TPinv[1][1] = r11 * r01 + 1.0
            TPinv[1][2] = r12 * r01
            TPinv[1][3] = -r01 * one_over_s
            TPinv[2][0] = r10 * r02
            TPinv[2][1] = r11 * r02
            TPinv[2][2] = r12 * r02 + 1.0
            TPinv[2][3] = -r02 * one_over_s
            TPinv[3][0] = -r10
            TPinv[3][1] = -r11
            TPinv[3][2] = -r12
            TPinv[3][3] = one_over_s

            _mat = Coordinate.preMult(TPinv, _mat)  # Finish computing full inverse of mat

        # undef px
        # undef py
        # undef pz
        # undef one_over_s
        # undef d
        else:  # Rightmost column is [0; 0; 0; 1] so it can be ignored
            r10 = mat[3][0]
            r11 = mat[3][1]
            r12 = mat[3][2]

            # Compute translation components of mat'
            _mat[3][0] = -(r10 * _mat[0][0] + r11 * _mat[1][0] + r12 * _mat[2][0])
            _mat[3][1] = -(r10 * _mat[0][1] + r11 * _mat[1][1] + r12 * _mat[2][1])
            _mat[3][2] = -(r10 * _mat[0][2] + r11 * _mat[1][2] + r12 * _mat[2][2])

        # undef tx
        # undef ty
        # undef tz
        return _mat

    @staticmethod
    def preMult(other, mat):
        t = []
        _mat = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for col in range(0, 4):
            t[0] = other[0][0] * mat[0][col] + other[0][1] * mat[1][col] + other[0][2] * mat[2][col] + other[0][3] * \
                   mat[3][col]
            t[1] = other[1][0] * mat[0][col] + other[1][1] * mat[1][col] + other[1][2] * mat[2][col] + other[1][3] * \
                   mat[3][col]
            t[2] = other[2][0] * mat[0][col] + other[2][1] * mat[1][col] + other[2][2] * mat[2][col] + other[2][3] * \
                   mat[3][col]
            t[3] = other[3][0] * mat[0][col] + other[3][1] * mat[1][col] + other[3][2] * mat[2][col] + other[3][3] * \
                   mat[3][col]
            _mat[0][col] = t[0]
            _mat[1][col] = t[1]
            _mat[2][col] = t[2]
            _mat[3][col] = t[3]
        return _mat

    @staticmethod
    def invert_4x4(mat):
        print('----------invert_4x4----------')
        indxc = [4]
        indxr = [4]
        ipiv = [4]

        # copy in place this may be unnecessary
        _mat = mat

        for j in range(0, 4):
            ipiv[j] = 0

        for i in range(0, 4):
            big = 0.0
            for j in range(0, 4):
                if ipiv[j] != 1:
                    for k in range(0, 4):
                        if ipiv[k] == 0:
                            if math.fabs(_mat[j][k]) >= big:
                                big = math.fabs(_mat[j][k])
                                irow = j
                                icol = k
                        elif ipiv[k] > 1:
                            print('false')
                            return
            ++(ipiv[icol])
            if irow != icol:
                for l in range(0, 4):
                    temp = _mat[irow][l]
                    _mat[irow][l] = _mat[icol][l]
                    _mat[icol][l] = temp

            indxr[i] = irow
            indxc[i] = icol
            if _mat[icol][icol] == 0:
                print('false')
                return

            pivinv = 1.0 / _mat[icol][icol]
            _mat[icol][icol] = 1
            for l in range(0, 4):
                _mat[icol][l] *= pivinv
            for ll in range(0, 4):
                if ll != icol:
                    dum = _mat[ll][icol]
                    _mat[ll][icol] = 0
                    for l in range(0, 4):
                        _mat[ll][l] -= _mat[icol][l] * dum

        for lx in range(4, 0):
            if indxr[lx - 1] != indxc[lx - 1]:
                for k in range(0, 4):
                    temp = _mat[k][indxr[lx - 1]]
                    _mat[k][indxr[lx - 1]] = _mat[k][indxc[lx - 1]]
                    _mat[k][indxc[lx - 1]] = temp

        return _mat

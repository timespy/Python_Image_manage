import cv2
import numpy as np
from PIL import Image
import random
import math
import os

DIR_SRC = "./src/sharp/"
DIR_DEST = "./dest/blur/"


def Simulation_route(T):
    X_shake = []
    Y_shake = []

    t = 0

    # x 方向上的振动仿真
    while t < T:
        shake_t = random.randint(0, 5)
        start = t
        t += shake_t
        if t > T:
            t = T
        A = random.randint(0, 20)
        w = random.randint(1, 10)
        X_shake.append([start, shake_t, A, w])

    t = 0
    # y 方向上的振动仿真
    while t < T:
        shake_t = random.randint(0, 5)
        start = t
        t += shake_t
        if t > T:
            t = T
        A = random.randint(0, 5)
        w = random.randint(1, 10)
        Y_shake.append([start, shake_t, A, w])

    shake = []
    len_x = len(X_shake)
    len_y = len(Y_shake)
    len_s = len_x + len_y

    i = 0
    j = 0
    for k in range(len_s):
        if i != len_x and j != len_y:
            if X_shake[i][0] <= Y_shake[j][0]:
                shake.append(['x', X_shake[i]])
                i += 1
            else:
                shake.append(['y', Y_shake[j]])
                j += 1
        elif j != len_y:
            shake.append(['y', Y_shake[j]])
            j += 1
        else:
            shake.append(['x', X_shake[i]])
            i += 1

    return shake


def Simulation_hormonic_matrix(img, width, height, T, A1, A2, w1, w2):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    ret_b = np.copy(b)
    ret_g = np.copy(g)
    ret_r = np.copy(r)

    for t in range(T):
        delX = int(A1 * math.cos(w1 * t))
        delY = int(A2 * math.cos(w2 * t))
        abs_X = math.fabs(delX)
        abs_Y = math.fabs(delY)
        trans_X = np.eye(width, dtype=np.float)
        trans_Y = np.eye(height, dtype=np.float)

        if delX > 0:
            for i in range(delX, width):
                for j in range(i-delX+1, i+1):
                    trans_X[j][i] = 1/abs_X
        else:
            for i in range(0, width+delX):
                for j in range(i, i-delX):
                    trans_X[j][i] = 1/abs_X
        if delY > 0:
            for i in range(delY, height):
                for j in range(i-delY+1, i+1):
                    trans_Y[i][j] = 1/abs_Y
        else:
            for i in range(0, height+delY):
                for j in range(i, i-delY):
                    trans_Y[i][j] = 1/abs_Y
        ret_b = np.dot(trans_Y, ret_b)
        ret_b = np.dot(ret_b, trans_X)
        ret_g = np.dot(trans_Y, ret_g)
        ret_g = np.dot(ret_g, trans_X)
        ret_r = np.dot(trans_Y, ret_r)
        ret_r = np.dot(ret_r, trans_X)
    ret_b = ret_b.astype(np.uint8)
    ret_g = ret_g.astype(np.uint8)
    ret_r = ret_r.astype(np.uint8)
    ret = cv2.merge([ret_b, ret_g, ret_r])
    return ret


def Simulation_blur(img):
    height, width, dim = img.shape
    ret = img
    route = Simulation_route(10)
    lens = len(route)
    pre_x_A = 0
    pre_x_w = 0
    pre_y_A = 0
    pre_y_w = 0
    f = route[0][0]
    if f == 'x':
        pre_x_A = route[0][1][2]
        pre_x_w = route[0][1][3]
    else:
        pre_y_A = route[0][1][2]
        pre_y_w = route[0][1][2]

    for i in range(0, lens-1):
        A = route[i][1][2]
        w = route[i][1][3]
        start = route[i][1][0]
        next_start = route[i+1][1][0]
        lasting = next_start - start
        flag = route[i][0]

        if flag == 'x':
            pre_x_A = A
            pre_x_w = w
            ret = Simulation_hormonic_matrix(ret, width, height, lasting, A, pre_y_A, w, pre_y_w)
        else:
            pre_y_A = A
            pre_y_w = w
            ret = Simulation_hormonic_matrix(ret, width, height, lasting, pre_x_A, A, pre_x_w, w)
    return ret


def main():
    # 检查源文件夹
    if not os.path.exists(DIR_SRC):
        os.mkdir(DIR_SRC)
    # 检查目标文件夹
    if not os.path.exists(DIR_DEST):
        os.mkdir(DIR_DEST)
    img = os.listdir(DIR_SRC)
    img_no = 1
    img_num = len(img)
    for i in img:
        im = cv2.imread(DIR_SRC + i)
        ret = Simulation_blur(im)
        name = i.split(".")
        cv2.imwrite(DIR_DEST + name[0] + '.png', ret)
        print('\rcompleted {}/{}'.format(img_no, img_num), end='')
        img_no += 1


if __name__ == '__main__':
    main()

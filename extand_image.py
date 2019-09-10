import torch
from torchvision import transforms
import torchvision
from PIL import Image
import torch.nn as nn
import time


def extand(param):
    C, H, W = param.shape
    extant_w = torch.zeros((H, W*2))
    extant_h = torch.zeros((H*2, W))
    for i in range(H):
        extant_w[i][2*i] = 1
        extant_w[i][2*i+1] = 1
    for i in range(W):
        extant_h[2*i][i] = 1
        extant_h[2*i+1][i] = 1
    ret = torch.zeros((C, H*2, W*2))
    for c in range(C):
        channel = torch.mm(param[c], extant_w)
        channel = torch.mm(extant_h, channel)
        ret[c] = channel
    return ret


def main():
    trans = transforms.Compose([transforms.ToTensor()])
    img_path = '1.png'
    img = Image.open(img_path).convert('RGB')
    test = trans(img)

    # test = torch.rand((3, 5, 5))
    # print(test)

    # test = torch.rand((3, 3))
    # print(test)

    print('start')
    start = time.time()
    res = extand(test)
    stop = time.time()
    torchvision.utils.save_image(res.data, './src/test/1_resize.png')
    print("time : {}".format(stop - start))

    # print(res)


if __name__ == '__main__':
    main()

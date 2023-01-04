# -*- coding: UTF-8 -*-
"""
@Project ：ZZYT
@File ：DataSet.py
@Author ：正途皆是道
@Date ：22-1-5 下午1:47
"""

import os
from paddle.io import Dataset
from paddle.io import DataLoader
import random
import numpy as np
from PIL import Image
from paddle.vision import transforms
from ConfigArgparse import parse_args

args = parse_args()


class ReadData(object):
    def __init__(self):
        """读取图像信息"""
        super(ReadData, self).__init__()

    def get_imgpath_label(self, label_file, img_dir, scale=1):
        # 获取图像和标签,生成交叉验证集8:2
        """返回图片绝对路径和对应的标签"""
        labels, img_paths = [], []
        with open(label_file, 'r') as f_dir:
            lines = f_dir.readlines()
        np.random.shuffle(lines)  # 打乱顺序
        for line in lines:
            line = line.strip()
            line_list = line.split('\t')
            img_paths.append(os.path.join(img_dir, line_list[0]))  # 绝对路径
            labels.append(int(line_list[1]))
        if scale == 1:  # 所有数据均为训练数据
            return labels, img_paths
        split_flag = int(len(labels) * scale)
        train_lab, eval_lab = labels[:split_flag], labels[split_flag:]
        train_img, eval_img = img_paths[:split_flag], img_paths[split_flag:]
        return train_lab, train_img, eval_lab, eval_img

    def get_dir_img_paths(self, img_dir):
        # 返回绝对路径
        result_list_1 = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
        result_list_1.sort(key=lambda x: int(x.split('/')[-1]))  # 排序
        if os.path.isdir(result_list_1[0]):  # result_list_1是文件夹
            result_list_2 = []
            for dir_path in result_list_1:
                file_list_t = os.listdir(dir_path)
                file_list_t.sort(key=lambda x: int(x.split('_')[0]))  # 排序
                result_list_2 += [os.path.join(dir_path, name) for name in file_list_t]
            return result_list_2
        return result_list_1


class DatasetAug():
    def __init__(self, mean, std):
        """数据增强"""
        # mean=[127.5, 127.5, 127.5]或者 11.0
        # std=[127.5, 127.5, 127.5] 或者 11.0 每个通道独立使用或者统一一个值
        self.mean = mean
        self.std = std

    def augop(self, pil_img, p):
        """如果给的是路径，则需要读取图片"""
        if type(pil_img) == str:
            pil_img = Image.open(pil_img)
        if pil_img.mode == 'L':
            pil_img = pil_img.convert('RGB')
        """亮度、对比度、饱和度和色调"""
        if random.random() < p:
            pil_img = transforms.adjust_brightness(pil_img, random.randint(5, 15) / 10.0)  # 图片随机增强0.5到1.5倍亮度
        if random.random() < p:
            pil_img = transforms.adjust_contrast(pil_img, random.randint(5, 15) / 10.0)  # 图片随机增强0.5到1.5倍对比度
        if random.random() < p:
            pil_img = transforms.adjust_hue(pil_img, random.randint(-1, 1) / 10.0)  # 图像的色调通道的偏移量最大范围[-0.5,0.5]
        if random.random() < p:
            pil_img = transforms.SaturationTransform(random.randint(0, 2) / 10.0)(pil_img)  # 调整图像的饱和度
        if random.random() < p:
            pil_img = transforms.HueTransform(random.randint(0, 2) / 10.0)(pil_img)  # 调整图像的色调。
        # pil_img = transforms.ColorJitter(random.randint(0, 3) / 10.0, random.randint(0, 3) / 10.0,
        #                                  random.randint(0, 3) / 10.0, random.randint(0, 3) / 10.0,
        #                                  keys=None)(pil_img)  # 随机调整图像的亮度，对比度，饱和度和色调。

        """裁剪，resize"""
        if random.random() < p:
            pil_img = transforms.resize(pil_img, (args.INPUT_SIZE[0] + 10, args.INPUT_SIZE[1] + 10),
                                        interpolation='bilinear')  # 将输入数据调整为指定大小。
            pil_img = transforms.RandomCrop(args.INPUT_SIZE)(pil_img)  # 在随机位置裁剪输入的图像，先将图像进行resize，保证尽量多的保留信息
        """图像翻转"""
        # pil_img = transforms.hflip(pil_img)  # 对输入图像进行水平翻转。
        # pil_img = transforms.vflip(pil_img)  # 对输入图像进行垂直方向翻转。
        pil_img = transforms.RandomHorizontalFlip(p)(pil_img)  # 基于概率来执行图片的水平翻转。
        pil_img = transforms.RandomVerticalFlip(p)(pil_img)  # 基于概率来执行图片的垂直翻转。
        """图像旋转"""
        if random.random() < p:
            pil_img = transforms.RandomRotation(90)(pil_img)  # 依据参数随机产生一个角度对图像进行旋转。
        if random.random() < p:
            pil_img = transforms.rotate(pil_img, 45)  # 按角度旋转图像
        """图像归一化"""
        # if random.random() < p:
        #     to_rgb = random.choice([False, True])
        #     pil_img = transforms.normalize(pil_img, self.mean, self.std, data_format='HWC',to_rgb=to_rgb) # 图像归一化处理
        pil_img = transforms.normalize(pil_img, self.mean, self.std, data_format='HWC')  # 图像归一化处理,to_rgb=to_rgb
        """调整图片大小"""
        pil_img = transforms.resize(pil_img, args.INPUT_SIZE, interpolation='bilinear')  # 将输入数据调整为指定大小。
        """图像维度置换,展示时注释掉本操作"""
        pil_img = transforms.Transpose(order=(2, 0, 1))(pil_img)  # 将输入的图像数据更改为目标格式 HWC -> CHW
        return pil_img.astype('float32')


class LoadTensorImg(Dataset):
    def __init__(self, imgs_path=[], label_list=[], hw_scale=(1080, 1920), dataaug=None, model='2train'):
        """载入图像数据"""
        super(LoadTensorImg, self).__init__()
        self.model = model
        self.imgs_path = imgs_path
        self.labels = label_list
        self.hw = hw_scale
        self.data_len = len(imgs_path)
        '''数据增强'''
        self.dataaug = dataaug
        self.aug_p = args.aug_p

    def __getitem__(self, item):
        if self.model == '2train':  # 仅训练时进行数据增强
            # image = process_image(self.imgs_path[item], self.model)
            image = self.dataaug.augop(self.imgs_path[item], self.aug_p)  # paddle提供的数据增强库
        else:  # 预测和评估时不需要增强,普通归一化即可
            image = self.img_to_chw_rgb(self.imgs_path[item])

        # image = paddle.to_tensor(image)  # CHW
        if self.model == '2pre':  # 预测时不提供label信息len(self.labels) == 0
            return image, self.imgs_path[item]

        if self.model == '2eval':  # 评估数据
            return image, self.labels[item], self.imgs_path[item]

        # label = paddle.to_tensor(self.labels[item])  # 参考1,2,3,4,5,loss,accuracy的输入维度[intN]
        return image, np.array([self.labels[item]])  # 训练数据 data,label

    def __len__(self):
        return self.data_len

    def img_to_chw_rgb(self, img_path):
        pil_img = Image.open(img_path)
        if pil_img.mode == 'L':
            pil_img = pil_img.convert('RGB')
        pil_img = transforms.resize(pil_img, args.INPUT_SIZE, interpolation='bilinear')  # 将输入数据调整为指定大小。
        pil_img = transforms.normalize(pil_img, args.rgb_mean, args.rgb_std, data_format='HWC')  # 归一化
        pil_img = transforms.Transpose(order=(2, 0, 1))(pil_img)  # 将输入的图像数据更改为目标格式 HWC -> CHW
        pil_img = pil_img.astype('float32')
        return pil_img


class LoadTensorNpy(Dataset):
    def __init__(self, npy_path, net_op='2train'):
        """载入.npy数据"""
        super(LoadTensorNpy, self).__init__()
        self.data_list = np.load(npy_path)
        self.data_len = len(self.data_list)
        self.net_op = net_op

    def __getitem__(self, item):
        if self.net_op == '2predect':
            # 预测时无需标签
            inputs_pre = self.data_list[item].astype('float32')
            inputs_pre = inputs_pre.reshape((1, 561))
            return inputs_pre
        inputs = self.data_list[item][1:]
        # 来一波数据增强（6，18，18）
        aug_random = random.random()
        if aug_random > 0.6 and self.net_op == '2train':  # 2eval时无需增强
            aug_index = random.randint(0, 560)
            inputs[aug_index] = inputs[aug_index] * aug_random
        label = self.data_list[item][0]
        inputs = inputs.reshape((1,561))#CL,C是通道数，L是特征长度
        return inputs.astype('float32'), np.array([label]).astype('int64')  # 训练数据 data,label

    def __len__(self):
        return self.data_len


def test_npy():
    train_dataset = LoadTensorNpy("/sdb1/数据/公共数据/中文新闻文本标题/test_np")
    inputs, label = train_dataset.__getitem__(10)
    print(inputs.shape)
    print(label)

    loader_train = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=False, num_workers=0)
    for (inputs, label) in loader_train:
        print(np.reshape(label.numpy(), (-1)), inputs.shape)


def test_img():
    dataaug = DatasetAug(args.rgb_mean, args.rgb_std)
    predata = ReadData()
    labels, imgs_path = predata.get_imgpath_label(args.train_label, args.train_dir)
    # labels, imgs_path = predata.get_imgpath_label(args.dev_label, args.dev_dir)
    train_dataset = LoadTensorImg(imgs_path=imgs_path, label_list=labels, hw_scale=args.INPUT_SIZE, dataaug=dataaug,
                                  model='2train')
    inputs, label = train_dataset.__getitem__(10)
    print(inputs.shape)
    print(label)
    loader_train_t = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=False)
    for (inputs, label) in loader_train_t:
        print(np.reshape(label.numpy(), (-1)), inputs.shape)


if __name__ == "__main__":
    test_img()
    # test_npy()

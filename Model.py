# -*- coding: UTF-8 -*-
"""
@Project ：ZZYT
@File ：Model.py
@Author ：正途皆是道
@Date ：22-1-5 下午1:46
"""
from paddle import nn
from paddle.vision import models
from ConfigArgparse import parse_args
import paddle

args = parse_args()

class NeuralNetworkStructure(nn.Layer):
    def __init__(self):
        """模型结构堆叠"""
        super(NeuralNetworkStructure, self).__init__()
        # 参数with_pool产生The fisrt matrix width should be same as second matrix height,
        # but received fisrt matrix width
        # self.base = models.MobileNetV2(num_classes=12)
        # self.base = models.ResNet(BasicBlock, 18)
        self.base = models.ResNet(models.resnet.BottleneckBlock, depth=50, num_classes=args.num_classes, with_pool=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.tanhshrink = nn.Tanhshrink()
        # x.shape:(6,18,18)
        self.conv1 = nn.Conv1D(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv1D(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv1D(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv1D(in_channels=64, out_channels=3, kernel_size=3)



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = paddle.tile(x, repeat_times=[553])
        x = x.reshape((-1,3,553,553))
        x = self.base(x)#NCHW
        # x = self.softmax(x)
        x = self.tanhshrink(x)
        # x = self.tanh(x)
        return x
        
modMODE_dict = {
            'NeuralNetworkStructure':NeuralNetworkStructure(),
        }

# -*- coding: UTF-8 -*-
"""
@Project ：ZZYT
@File ：ConfigArgparse.py
@Author ：正途皆是道
@Date ：22-1-5 下午1:54
"""
import argparse

        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='NeuralNetworkStructure', help='模型结构名称')
    parser.add_argument('--model_id', type=int, default=-1, help='预训练模型数字id，-1时不使用预训练')
    parser.add_argument('--save_train_model_dir', type=str, default='save_model/train', help='训练模型保存文件夹,不以/结尾')
    parser.add_argument('--save_train_best_model', type=str, default='save_model/best', help='acc最大的模型保存文件夹,不以/结尾')
    parser.add_argument('--save_infer_model_dir', type=str, default='save_model/infer', help='推理模型保存文件夹,不以/结尾')
    parser.add_argument('--gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--rgb_mean', type=tuple, default=(67.0, 33.0, 11.0), help='数据平均值')
    parser.add_argument('--rgb_std', type=tuple, default=(75.0, 40.0, 19.0), help='数据方差')
    parser.add_argument('--aug_p', type=float, default=0.3, help='数据增强概率')
    parser.add_argument('--num_classes', type=int, default=6, help='模型输出类别')
    parser.add_argument('--label_text', type=str, default='/ing/GAMMA.txt', help='标签文件')
    parser.add_argument('--train_label', type=str, default='/yhdfev_jpg_one/voc_train.txt', help='标签文件')
    parser.add_argument('--train_dir', type=str, default='/sdbhdfev_jpg_one', help='训练数据,npy或者图片')
    parser.add_argument('--dev_label', type=str, default='/sdb1dfhv_jpg_one/voc_train.txt', help='标签文件')
    parser.add_argument('--dev_dir', type=str, default='/sdb1dfhdev_jpg_one', help='训练评估数据,格式和训练图片一直')
    parser.add_argument('--test_dir', type=str, default='/homedhdt_jpg_one', help='测试数据,npy或者图片')
    parser.add_argument('--INPUT_SIZE', type=tuple, default=(64, 64), help='模型输入尺寸(448, 448)')
    parser.add_argument('--BATCH_SIZE', type=int, default=25, help='训练批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='learning_rate学习率')
    parser.add_argument('--label_dict', type=dict, default={'LAYING': 0, 'STANDING': 1, 'SITTING': 2, 'WALKING': 3, 'WALKING_UPSTAIRS': 4,
                      'WALKING_DOWNSTAIRS': 5}, help='标签字典')
    return parser.parse_args()


args_dict = vars(parse_args())
for args_p in zip(list(args_dict.keys()), list(args_dict.values())):
    print('{:>30}│ {}'.format(args_p[0], args_p[1]))
    print('─'*30+'┼'+'─'*50)

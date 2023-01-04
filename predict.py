# -*- coding: UTF-8 -*-
"""
@Project ：ZZYT
@File ：EvalPredect.py
@Author ：正途皆是道
@Date ：22-1-5 下午4:50
"""
import csv
import os
import paddle
from paddle.io import DataLoader
import numpy as np

from Model import modMODE_dict
from DataSet import  LoadTensorNpy
from ConfigArgparse import parse_args

args = parse_args()


# 评估和预测
class EP(object):
    def predect(self):
        model = modMODE_dict[args.model_name]
        model_state_dict = paddle.load('{}'.format(args.save_train_best_model))
        model.load_dict(model_state_dict)
        model.eval()  # 不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False

        def loader_predect_npy_data():
            # npy作为输入
            test_dataset = LoadTensorNpy('data/data137267/test.npy', '2predect')
            # 导入TensorData
            loader_test = DataLoader(test_dataset, batch_size=args.BATCH_SIZE)
            return loader_test, test_dataset.data_len



        loader_test, test_len = loader_predect_npy_data()
        # loader_test, test_len = loader_predect_img_data()
        label_list = list(args.label_dict.keys())
        with open('submission.csv', 'w+', encoding='utf-8', newline='') as csf:
            writer = csv.writer(csf)
            writer.writerow(['Activity'])
            for batch_id, inputs in enumerate(loader_test()):
                predicts = model(inputs)
                pre_np = predicts.numpy()
                for result_i in range(pre_np.shape[0]):
                    lable_indx = np.argmax(pre_np[result_i])
                    label_txt = label_list[lable_indx]
                    writer.writerow([label_txt])

                print('\r{:>5}/{}'.format(args.BATCH_SIZE * (batch_id + 1), test_len), end='', flush=True)

if __name__ == '__main__':
    EP().predect()

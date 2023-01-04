# -*- coding: UTF-8 -*-
"""
@Project ：zzytgitee
@File ：Test_20210806.py
@Author ：正途皆是道
@Date ：21-8-6 上午9:08
"""
import time
import os
import paddle
from paddle.io import DataLoader
from paddle import nn
import numpy as np

from Model import modMODE_dict
from DataSet import ReadData, DatasetAug, LoadTensorImg, LoadTensorNpy
from ConfigArgparse import parse_args

args = parse_args()

# 二次训练导入参数模型
continue_id = args.model_id  # 首次次训练continue_id = None


# paddle.device.set_device("cpu") #在gpu环境下使用cpu


class app():
    def __init__(self):
        """训练、预测、评估"""
        super(app).__init__()

    def train(self):
        def del_models():
            # 删除模型，保留新的模型
            models_ctime = []
            model_dir = 'save_model/train'
            if os.path.isdir(model_dir):
                for name in os.listdir(model_dir):
                    model_path = os.path.join(model_dir, name)
                    ctime = os.path.getctime(model_path)  # 创建时间
                    models_ctime.append([model_path, ctime])
                models_ctime.sort(key=lambda x: x[1])
                for model_ctime in models_ctime[0:-2]:
                    # print('删除模型数据：', model_ctime[0])
                    os.system('rm -rf {}'.format(model_ctime[0]))

        def loader_train_npy_data():
            # 导入训练数据
            # npy作为输入
            train_dataset = LoadTensorNpy('data/data137267/正常值.npy', '2train')
            # 导入TensorData
            loader_train_t = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=False)

            # 导入deviation训练评估数据
            # npy作为输入
            dev_dataset = LoadTensorNpy('data/data137267/正常值.npy', '2eval')
            # 导入TensorData
            loader_dev_t = DataLoader(dev_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=False)

            return loader_train_t, train_dataset.data_len, loader_dev_t, dev_dataset.data_len

        def loader_train_img_data():
            # 导入Img训练数据
            dataaug = DatasetAug(args.rgb_mean, args.rgb_std)
            predata = ReadData()
            labels, imgs_path = predata.get_imgpath_label(args.train_label, args.train_dir)
            train_dataset = LoadTensorImg(imgs_path=imgs_path, label_list=labels, hw_scale=args.INPUT_SIZE,
                                          dataaug=dataaug, model='2train')
            loader_train_t = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=False)
            # 导入deviation训练评估数据
            labels, imgs_path = predata.get_imgpath_label(args.dev_label, args.dev_dir)
            dev_dataset = LoadTensorImg(imgs_path=imgs_path, label_list=labels, hw_scale=args.INPUT_SIZE,
                                        dataaug=dataaug, model='2train')
            loader_dev_t = DataLoader(dev_dataset, batch_size=args.BATCH_SIZE, drop_last=False)
            return loader_train_t, train_dataset.data_len, loader_dev_t, dev_dataset.data_len

        loader_train, train_len, loader_dev, dev_len = loader_train_npy_data()
        # loader_train, train_len, loader_dev, dev_len = loader_train_img_data()
        train_batch_all = train_len // args.BATCH_SIZE  # 总的需要多少个批次
        model = modMODE_dict[args.model_name]
        model.train()  # 启用BatchNormalization和 Dropout，将BatchNormalization和Dropout置为True
        opt_function = paddle.optimizer.Adam(learning_rate=args.lr, beta1=0.9, beta2=0.999, parameters=model.parameters())
        loss_function = nn.CrossEntropyLoss()  # 多label，soft_label=True
        if continue_id != -1:  # 二次训练，读取保存的参数
            layer_state_dict = paddle.load('{}/train_{}.pdparams'.format(args.save_train_model_dir, continue_id))
            opt_state_dict = paddle.load('{}/train_{}.pdopt'.format(args.save_train_model_dir, continue_id))
            model.set_state_dict(layer_state_dict)
            opt_function.set_state_dict(opt_state_dict)
        acc_max = 0.
        best_model_id = -1
        os.system('rm -rf ./best_acc.txt')
        for epoch_id in range(20000):
            # 开始训练
            tp_num, in_num = 0, 0
            epoch_t_start = time.time()
            for batch_id, (image, label) in enumerate(loader_train()):
                # print(np.reshape(label.numpy(),(-1)),image.shape)
                epoch_b_start = time.time()
                out = model(image)
                loss = loss_function(out, label)
                # acc = paddle.metric.accuracy(out, label)
                epoch_b_end = time.time()

                for i in range(args.BATCH_SIZE):  # 计算全局正确率
                    try:
                        predict = np.argmax(out.numpy()[i])
                        gt = label.numpy()[i]
                        if gt == predict:
                            tp_num += 1
                        in_num += 1
                    except:
                        '''不是一个完整的BATCH_SIZE'''

                if batch_id % 20 == 0:
                    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
                    print("time:{},train: epoch:{:>3d}, epoch_time:{:>6.3f}, batch_id:{:>2d}/{}, batch_time:{:>5.3f},"
                          " loss is:{:>5.3f}, train_accuracy:{:>5.3f}".format(now_time, epoch_id,
                                                                              epoch_b_end - epoch_t_start,
                                                                              batch_id, train_batch_all,
                                                                              epoch_b_end - epoch_b_start,
                                                                              loss.numpy()[0], tp_num / in_num))
                loss.backward()
                opt_function.step()
                opt_function.clear_grad()
            # 删除模型
            # if epoch_id % 10 == 0:
            #     del_models()
            if epoch_id % 10 == 0:
                # 模型保存
                paddle.save(model.state_dict(), '{}/train_{}.pdparams'.format(args.save_train_model_dir, epoch_id))
                paddle.save(opt_function.state_dict(), '{}/train_{}.pdopt'.format(args.save_train_model_dir, epoch_id))
                # 模型评估
                acc_list = []
                for batch_id, (image, label) in enumerate(loader_dev()):
                    out = model(image)
                    acc = paddle.metric.accuracy(out, label)
                    acc_list.append(acc.numpy()[0])
                epoch_time = (time.time() - epoch_b_start) / 60
                acc_mean = sum(acc_list) / len(acc_list)
                if acc_max < acc_mean:
                    acc_max = acc_mean
                    best_model_id = epoch_id
                    # 保存最优模型
                    paddle.save(model.state_dict(), f'{args.save_train_best_model}/best.pdparams')
                    paddle.save(opt_function.state_dict(), f'{args.save_train_best_model}/best.pdopt')
                    with open('./best_acc.txt', 'a+', encoding='utf-8') as best_acc:
                        best_acc.write(f'{acc_max}\r\n')
                print('epoch_id：{},acc：{} ,acc_max：{} ,best_model_id：{} time：{:.2f}'.format(epoch_id, acc_mean, acc_max,
                                                                                            best_model_id, epoch_time))
            

if __name__ == '__main__':
    # 图像分类
    # net = modMODE_dict[args.model_name]
    # paddle.summary(net, (20, 3, INPUT_SIZE[0], INPUT_SIZE[1]))
    app().train()  # 由全局变量continue_id确定是否导入模型参数

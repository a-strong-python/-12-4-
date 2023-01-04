# 一、竞赛介绍
[比赛的地址](https://aistudio.baidu.com/aistudio/competition/detail/245/0/introduction)
    本赛题以智能手机识别人类行为为背景，要求选手根据手机识别的数据对人类行为进行预测。这是一个典型的分类问题，属于结构化数据挖掘赛题。<br>
    实验在19-48岁年龄段的30名志愿者中进行，每个人在腰部佩戴某品牌的智能手机进行六项活动（步行、楼上步行、楼下步行、坐、站、躺），实验以50Hz的恒定速率捕获3轴线性加速度和3轴角速度。<br>
1.1 提交内容及格式<br>
1. 本次比赛要求参赛选手必须使用飞桨（PaddlePaddle）深度学习框架 训练的模型；
2. 结果文件命名：submission.zip；
3. 结果文件格式：zip文件格式，zip文件解压后为1个submission.csv文件，编码为UTF-8；
4. 结果文件内容：submission.csv仅包含1个字段，为Activity字段：
5. 提交示例：
| Activity 1 | 
| --------  | 
| STANDING  |
LAYING
WALKING
SITTING
WALKING
…
WALKING_DOWNSTAIRS




# 二、数据处理
## 2.1、导入相关包


```python
# -*- coding: UTF-8 -*-
"""
@项目名称：手机行为预测.py
@作   者：陆地起飞全靠浪
@创建日期：2022-05-10-10:32
"""
import csv
import glob

import unittest
import cv2 as cv
import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cbook import boxplot_stats
```

## 2.2、定义全局变量


```python
parent_path = 'data/data137267'
# 将6项标签存入字典中：步行、楼上步行、楼下步行、坐、站、躺
label_dict = {'LAYING': 0, 'STANDING': 1, 'SITTING': 2, 'WALKING': 3, 'WALKING_UPSTAIRS': 4,'WALKING_DOWNSTAIRS': 5}
label_keys = list(label_dict.keys())
```

## 2.3、解压数据


```python
os.system(f'unzip -d {parent_path}/ -oq {parent_path}/train.csv.zip')
os.system(f'unzip -d {parent_path}/ -oq {parent_path}/test.csv.zip')
```




    0



## 2.4、读取CSV数据，按标签返回字典


```python
def read_csv_as_dict(train_csv_path):
    # 用于去除训练集中的异常值
    data_dict = {}
    with open(train_csv_path, encoding='utf-8') as csf:
        read_train_csv = csv.reader(csf)
        head = next(read_train_csv)
        for row in read_train_csv:
            key = row[-1]
            value_str = row[0:-1]  # len(value_str)=561
            value_float = [float(x) for x in value_str]
            # if min(value_float) <-1 or max(value_float)>20:
            #     continue
            try:
                data_dict[key] += value_float
            except:
                data_dict[key] = value_float

                # print(row)
    return data_dict
```

## 2.5、通过箱线图去除训练集中每个标签中的异常值,结果保存为npy矩阵
#### 2.5.1、绘制箱线图


```python
# 画箱线图
train_csv_path = f'{parent_path}/train.csv'
test_csv_path = f'{parent_path}/test.csv'

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# print(train_df.shape)
# print(train_df.columns)
# 统计Activity数量并画图
# train_df['Activity'].value_counts().plot(kind='bar')
plt.figure(figsize=(20, 5))
box_width = 0.5
plot_x = 'Activity'
plot_y = 'tBodyAcc-mean()-X'
ax = sns.boxplot(data=train_df, y=plot_y, x=plot_x, width=box_width)
i = 0
for name, group in train_df.groupby(plot_x):
    Q0, Q1, Q2, Q3, Q4 = group[plot_y].quantile([0, 0.25, 0.5, 0.75, 1])
    for q in (Q0, Q1, Q2, Q3, Q4):
        x = i - box_width / 2
        y = q
        ax.annotate('%.2f' % q, (x, y),
                    xytext=(x - 0.1, y), textcoords='data',
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    va='center', ha='right')

    i += 1
plt.tight_layout()
plt.savefig('pltSavefig.jpg')
```


    
![png](output_10_0.png)
    


#### 2.5.2、去除异常值


```python
# 去除异常值
train_csv_path = f'{parent_path}/train.csv'
data_dict = read_csv_as_dict(train_csv_path)
# 去除每个参数中的异常值，561个参数
for label_key in label_keys:
    inputx_list = data_dict[label_key]
    inputx_np = np.array(inputx_list)
    inputx_561 = inputx_np.reshape((-1, 561))  # 每行仅有561个参数
    inputx_trans = inputx_561.transpose(1, 0)  # 行列互换(561,-1)
    输入量 = inputx_trans.shape[1]
    # 通过箱线图whislo、whishi去除异常值
    outliers_index = []  # 所有异常值索引
    for inputx in inputx_trans:  # 对561个参数绘制箱线图

        df = pd.DataFrame(data={'x_axis': [label_key] * inputx_trans.shape[1],
                                'y_axis': inputx})
        # 获取四线值
        lo_whisker = boxplot_stats(df.y_axis)[0]['whislo']  # 最小值
        hi_whisker = boxplot_stats(df.y_axis)[0]['whishi']  # 最大值
        q1 = boxplot_stats(df.y_axis)[0]['q1']  # 1/4
        q3 = boxplot_stats(df.y_axis)[0]['q3']  # 3/4
        small_outliers_index = np.where(inputx < lo_whisker)[0].tolist()  # 较小异常值索引
        large_outliers_index = np.where(inputx > hi_whisker)[0].tolist()  # 较大异常值索引
        outliers_index += small_outliers_index
        outliers_index += large_outliers_index

    # 差集 所有的索引减去异常索引
    right_index = list(set(range(输入量)).difference(set(outliers_index)))
    right_inputx = inputx_trans[:, right_index]
    right_trans = right_inputx.transpose(1, 0)  # 行列互换(-1,561)
    # 创建等shape标签值
    label_np = np.ones((right_trans.shape[0], 1)).astype('int32') * label_dict[label_key]
    save_np = np.concatenate((label_np, right_trans), axis=1)
    print(right_trans.shape)
    # 保存
    np.save(train_csv_path[:-4] + f'_{label_key}.npy', save_np)

```

    (360, 561)
    (400, 561)
    (361, 561)
    (198, 561)
    (265, 561)
    (153, 561)


## 2.6、合并去除异常值后的npy结果


```python
npy_path_list = glob.glob(f'{parent_path}/*.npy')
save_np = np.load(npy_path_list[0])
for npy_path in npy_path_list[1:]:
    npy = np.load(npy_path)
    save_np = np.concatenate((save_np, npy), axis=0)
np.save(f'{parent_path}/正常值.npy', save_np)

```

# 三、训练


```python
# 继续训练从save_model/train中导入的模型编号
!python train.py  \
    --model_name NeuralNetworkStructure \
    --model_id 270 \
    --BATCH_SIZE 2
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
                        model_name│ NeuralNetworkStructure
    ──────────────────────────────┼──────────────────────────────────────────────────
                          model_id│ 270
    ──────────────────────────────┼──────────────────────────────────────────────────
              save_train_model_dir│ save_model/train
    ──────────────────────────────┼──────────────────────────────────────────────────
             save_train_best_model│ save_model/best
    ──────────────────────────────┼──────────────────────────────────────────────────
              save_infer_model_dir│ save_model/infer
    ──────────────────────────────┼──────────────────────────────────────────────────
                               gpu│ True
    ──────────────────────────────┼──────────────────────────────────────────────────
                          rgb_mean│ (67.0, 33.0, 11.0)
    ──────────────────────────────┼──────────────────────────────────────────────────
                           rgb_std│ (75.0, 40.0, 19.0)
    ──────────────────────────────┼──────────────────────────────────────────────────
                             aug_p│ 0.3
    ──────────────────────────────┼──────────────────────────────────────────────────
                       num_classes│ 6
    ──────────────────────────────┼──────────────────────────────────────────────────
                        label_text│ /ing/GAMMA.txt
    ──────────────────────────────┼──────────────────────────────────────────────────
                       train_label│ /yhdfev_jpg_one/voc_train.txt
    ──────────────────────────────┼──────────────────────────────────────────────────
                         train_dir│ /sdbhdfev_jpg_one
    ──────────────────────────────┼──────────────────────────────────────────────────
                         dev_label│ /sdb1dfhv_jpg_one/voc_train.txt
    ──────────────────────────────┼──────────────────────────────────────────────────
                           dev_dir│ /sdb1dfhdev_jpg_one
    ──────────────────────────────┼──────────────────────────────────────────────────
                          test_dir│ /homedhdt_jpg_one
    ──────────────────────────────┼──────────────────────────────────────────────────
                        INPUT_SIZE│ (64, 64)
    ──────────────────────────────┼──────────────────────────────────────────────────
                        BATCH_SIZE│ 2
    ──────────────────────────────┼──────────────────────────────────────────────────
                                lr│ 0.001
    ──────────────────────────────┼──────────────────────────────────────────────────
                        label_dict│ {'LAYING': 0, 'STANDING': 1, 'SITTING': 2, 'WALKING': 3, 'WALKING_UPSTAIRS': 4, 'WALKING_DOWNSTAIRS': 5}
    ──────────────────────────────┼──────────────────────────────────────────────────
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:653: UserWarning: When training, we now always track global mean and variance.
      "When training, we now always track global mean and variance.")
    time:2022-12-01 17:41:40,train: epoch:  0, epoch_time: 2.910, batch_id: 0/868, batch_time:2.907, loss is:0.000, train_accuracy:1.000
    time:2022-12-01 17:43:49,train: epoch:  0, epoch_time:132.493, batch_id:20/868, batch_time:2.188, loss is:0.000, train_accuracy:0.881
    time:2022-12-01 17:45:57,train: epoch:  0, epoch_time:260.173, batch_id:40/868, batch_time:2.192, loss is:0.000, train_accuracy:0.854
    time:2022-12-01 17:48:05,train: epoch:  0, epoch_time:388.772, batch_id:60/868, batch_time:2.194, loss is:0.000, train_accuracy:0.877
    time:2022-12-01 17:50:14,train: epoch:  0, epoch_time:517.768, batch_id:80/868, batch_time:2.185, loss is:0.000, train_accuracy:0.889
    time:2022-12-01 17:52:22,train: epoch:  0, epoch_time:645.708, batch_id:100/868, batch_time:2.265, loss is:0.000, train_accuracy:0.886
    time:2022-12-01 17:54:33,train: epoch:  0, epoch_time:775.966, batch_id:120/868, batch_time:2.272, loss is:0.004, train_accuracy:0.888
    time:2022-12-01 17:56:41,train: epoch:  0, epoch_time:904.783, batch_id:140/868, batch_time:2.146, loss is:7.146, train_accuracy:0.894
    time:2022-12-01 17:58:50,train: epoch:  0, epoch_time:1033.130, batch_id:160/868, batch_time:2.166, loss is:0.000, train_accuracy:0.907
    time:2022-12-01 18:00:57,train: epoch:  0, epoch_time:1160.252, batch_id:180/868, batch_time:2.201, loss is:0.518, train_accuracy:0.906
    time:2022-12-01 18:03:08,train: epoch:  0, epoch_time:1291.200, batch_id:200/868, batch_time:2.232, loss is:0.000, train_accuracy:0.913
    time:2022-12-01 18:05:15,train: epoch:  0, epoch_time:1418.782, batch_id:220/868, batch_time:2.186, loss is:0.001, train_accuracy:0.910
    time:2022-12-01 18:07:24,train: epoch:  0, epoch_time:1546.832, batch_id:240/868, batch_time:2.197, loss is:0.020, train_accuracy:0.902
    time:2022-12-01 18:09:32,train: epoch:  0, epoch_time:1674.833, batch_id:260/868, batch_time:2.128, loss is:0.001, train_accuracy:0.906
    time:2022-12-01 18:11:39,train: epoch:  0, epoch_time:1801.832, batch_id:280/868, batch_time:2.184, loss is:0.003, train_accuracy:0.911
    time:2022-12-01 18:13:48,train: epoch:  0, epoch_time:1931.527, batch_id:300/868, batch_time:2.351, loss is:0.000, train_accuracy:0.912
    time:2022-12-01 18:15:57,train: epoch:  0, epoch_time:2060.092, batch_id:320/868, batch_time:2.385, loss is:0.002, train_accuracy:0.916
    time:2022-12-01 18:18:04,train: epoch:  0, epoch_time:2187.721, batch_id:340/868, batch_time:2.264, loss is:0.552, train_accuracy:0.912
    time:2022-12-01 18:20:11,train: epoch:  0, epoch_time:2314.488, batch_id:360/868, batch_time:2.259, loss is:0.014, train_accuracy:0.916
    time:2022-12-01 18:22:19,train: epoch:  0, epoch_time:2442.255, batch_id:380/868, batch_time:2.216, loss is:0.004, train_accuracy:0.919
    time:2022-12-01 18:24:27,train: epoch:  0, epoch_time:2570.112, batch_id:400/868, batch_time:2.161, loss is:0.000, train_accuracy:0.921
    time:2022-12-01 18:26:34,train: epoch:  0, epoch_time:2697.663, batch_id:420/868, batch_time:2.144, loss is:0.000, train_accuracy:0.924
    time:2022-12-01 18:28:43,train: epoch:  0, epoch_time:2826.321, batch_id:440/868, batch_time:2.243, loss is:2.416, train_accuracy:0.923
    time:2022-12-01 18:30:53,train: epoch:  0, epoch_time:2956.041, batch_id:460/868, batch_time:2.243, loss is:0.000, train_accuracy:0.923
    time:2022-12-01 18:32:59,train: epoch:  0, epoch_time:3082.513, batch_id:480/868, batch_time:2.176, loss is:0.000, train_accuracy:0.924
    time:2022-12-01 18:35:07,train: epoch:  0, epoch_time:3209.964, batch_id:500/868, batch_time:2.215, loss is:0.000, train_accuracy:0.923
    time:2022-12-01 18:37:14,train: epoch:  0, epoch_time:3337.120, batch_id:520/868, batch_time:2.149, loss is:0.359, train_accuracy:0.924
    time:2022-12-01 18:39:23,train: epoch:  0, epoch_time:3465.873, batch_id:540/868, batch_time:2.153, loss is:0.215, train_accuracy:0.925
    time:2022-12-01 18:41:31,train: epoch:  0, epoch_time:3594.441, batch_id:560/868, batch_time:2.246, loss is:0.000, train_accuracy:0.925
    time:2022-12-01 18:43:39,train: epoch:  0, epoch_time:3721.994, batch_id:580/868, batch_time:2.206, loss is:0.000, train_accuracy:0.923
    time:2022-12-01 18:45:46,train: epoch:  0, epoch_time:3849.367, batch_id:600/868, batch_time:2.190, loss is:0.013, train_accuracy:0.921
    time:2022-12-01 18:47:53,train: epoch:  0, epoch_time:3976.647, batch_id:620/868, batch_time:2.144, loss is:0.000, train_accuracy:0.919
    time:2022-12-01 18:50:02,train: epoch:  0, epoch_time:4105.454, batch_id:640/868, batch_time:2.271, loss is:0.000, train_accuracy:0.921
    time:2022-12-01 18:52:11,train: epoch:  0, epoch_time:4234.260, batch_id:660/868, batch_time:2.175, loss is:0.000, train_accuracy:0.923
    time:2022-12-01 18:54:23,train: epoch:  0, epoch_time:4365.797, batch_id:680/868, batch_time:2.310, loss is:2.617, train_accuracy:0.922
    time:2022-12-01 18:56:31,train: epoch:  0, epoch_time:4494.572, batch_id:700/868, batch_time:2.304, loss is:0.000, train_accuracy:0.924
    time:2022-12-01 18:58:42,train: epoch:  0, epoch_time:4625.042, batch_id:720/868, batch_time:2.183, loss is:0.000, train_accuracy:0.924
    time:2022-12-01 19:00:51,train: epoch:  0, epoch_time:4753.997, batch_id:740/868, batch_time:2.241, loss is:0.000, train_accuracy:0.924
    time:2022-12-01 19:03:00,train: epoch:  0, epoch_time:4882.809, batch_id:760/868, batch_time:2.201, loss is:0.000, train_accuracy:0.926
    time:2022-12-01 19:05:12,train: epoch:  0, epoch_time:5015.403, batch_id:780/868, batch_time:2.393, loss is:0.000, train_accuracy:0.925
    time:2022-12-01 19:07:20,train: epoch:  0, epoch_time:5143.595, batch_id:800/868, batch_time:2.190, loss is:0.000, train_accuracy:0.926
    time:2022-12-01 19:09:28,train: epoch:  0, epoch_time:5271.593, batch_id:820/868, batch_time:2.238, loss is:0.000, train_accuracy:0.927
    time:2022-12-01 19:11:37,train: epoch:  0, epoch_time:5399.930, batch_id:840/868, batch_time:2.274, loss is:0.000, train_accuracy:0.928
    time:2022-12-01 19:13:45,train: epoch:  0, epoch_time:5528.001, batch_id:860/868, batch_time:2.251, loss is:0.000, train_accuracy:0.930
    epoch_id：0,acc：0.9781357882623706 ,acc_max：0.9781357882623706 ,best_model_id：0 time：32.64
    time:2022-12-01 19:47:15,train: epoch:  1, epoch_time: 2.252, batch_id: 0/868, batch_time:2.251, loss is:0.001, train_accuracy:1.000
    time:2022-12-01 19:49:24,train: epoch:  1, epoch_time:131.193, batch_id:20/868, batch_time:2.174, loss is:0.000, train_accuracy:1.000
    time:2022-12-01 19:51:32,train: epoch:  1, epoch_time:259.967, batch_id:40/868, batch_time:2.248, loss is:0.000, train_accuracy:0.988
    time:2022-12-01 19:53:41,train: epoch:  1, epoch_time:388.740, batch_id:60/868, batch_time:2.180, loss is:0.000, train_accuracy:0.992
    time:2022-12-01 19:55:51,train: epoch:  1, epoch_time:519.104, batch_id:80/868, batch_time:2.169, loss is:0.057, train_accuracy:0.988
    time:2022-12-01 19:57:58,train: epoch:  1, epoch_time:645.761, batch_id:100/868, batch_time:2.192, loss is:0.000, train_accuracy:0.980
    time:2022-12-01 20:00:07,train: epoch:  1, epoch_time:774.736, batch_id:120/868, batch_time:2.189, loss is:0.046, train_accuracy:0.983
    time:2022-12-01 20:02:18,train: epoch:  1, epoch_time:905.309, batch_id:140/868, batch_time:2.198, loss is:0.000, train_accuracy:0.986
    time:2022-12-01 20:04:27,train: epoch:  1, epoch_time:1035.100, batch_id:160/868, batch_time:2.213, loss is:0.000, train_accuracy:0.988
    time:2022-12-01 20:06:36,train: epoch:  1, epoch_time:1163.380, batch_id:180/868, batch_time:2.229, loss is:0.000, train_accuracy:0.986
    time:2022-12-01 20:08:43,train: epoch:  1, epoch_time:1291.131, batch_id:200/868, batch_time:2.171, loss is:0.000, train_accuracy:0.985
    time:2022-12-01 20:10:52,train: epoch:  1, epoch_time:1419.288, batch_id:220/868, batch_time:2.286, loss is:0.179, train_accuracy:0.986
    time:2022-12-01 20:12:59,train: epoch:  1, epoch_time:1547.062, batch_id:240/868, batch_time:2.209, loss is:0.002, train_accuracy:0.985
    time:2022-12-01 20:15:09,train: epoch:  1, epoch_time:1676.273, batch_id:260/868, batch_time:2.217, loss is:0.000, train_accuracy:0.985
    time:2022-12-01 20:17:17,train: epoch:  1, epoch_time:1804.906, batch_id:280/868, batch_time:2.208, loss is:0.000, train_accuracy:0.982
    time:2022-12-01 20:19:24,train: epoch:  1, epoch_time:1931.668, batch_id:300/868, batch_time:2.138, loss is:0.700, train_accuracy:0.980
    time:2022-12-01 20:21:32,train: epoch:  1, epoch_time:2059.626, batch_id:320/868, batch_time:2.212, loss is:0.001, train_accuracy:0.980
    time:2022-12-01 20:23:41,train: epoch:  1, epoch_time:2188.171, batch_id:340/868, batch_time:2.120, loss is:0.000, train_accuracy:0.978
    time:2022-12-01 20:25:50,train: epoch:  1, epoch_time:2317.267, batch_id:360/868, batch_time:2.319, loss is:0.163, train_accuracy:0.975
    time:2022-12-01 20:28:03,train: epoch:  1, epoch_time:2450.882, batch_id:380/868, batch_time:2.270, loss is:0.015, train_accuracy:0.976
    time:2022-12-01 20:30:11,train: epoch:  1, epoch_time:2579.037, batch_id:400/868, batch_time:2.217, loss is:0.000, train_accuracy:0.976
    time:2022-12-01 20:32:20,train: epoch:  1, epoch_time:2707.455, batch_id:420/868, batch_time:2.166, loss is:0.000, train_accuracy:0.977
    time:2022-12-01 20:34:28,train: epoch:  1, epoch_time:2835.833, batch_id:440/868, batch_time:2.184, loss is:0.000, train_accuracy:0.977
    time:2022-12-01 20:36:40,train: epoch:  1, epoch_time:2967.504, batch_id:460/868, batch_time:2.276, loss is:0.000, train_accuracy:0.977
    time:2022-12-01 20:38:54,train: epoch:  1, epoch_time:3101.585, batch_id:480/868, batch_time:2.301, loss is:0.000, train_accuracy:0.976
    time:2022-12-01 20:41:07,train: epoch:  1, epoch_time:3234.184, batch_id:500/868, batch_time:2.349, loss is:0.000, train_accuracy:0.977
    time:2022-12-01 20:43:14,train: epoch:  1, epoch_time:3361.446, batch_id:520/868, batch_time:2.187, loss is:0.000, train_accuracy:0.978
    time:2022-12-01 20:45:21,train: epoch:  1, epoch_time:3488.350, batch_id:540/868, batch_time:2.094, loss is:0.001, train_accuracy:0.977
    time:2022-12-01 20:47:27,train: epoch:  1, epoch_time:3614.718, batch_id:560/868, batch_time:2.327, loss is:2.449, train_accuracy:0.976
    time:2022-12-01 20:49:35,train: epoch:  1, epoch_time:3743.131, batch_id:580/868, batch_time:2.223, loss is:0.000, train_accuracy:0.975
    time:2022-12-01 20:51:46,train: epoch:  1, epoch_time:3873.842, batch_id:600/868, batch_time:2.136, loss is:0.000, train_accuracy:0.972
    time:2022-12-01 20:53:54,train: epoch:  1, epoch_time:4001.301, batch_id:620/868, batch_time:2.243, loss is:0.000, train_accuracy:0.972
    time:2022-12-01 20:56:02,train: epoch:  1, epoch_time:4130.099, batch_id:640/868, batch_time:2.229, loss is:0.000, train_accuracy:0.972
    time:2022-12-01 20:58:13,train: epoch:  1, epoch_time:4261.007, batch_id:660/868, batch_time:2.203, loss is:0.000, train_accuracy:0.972
    time:2022-12-01 21:00:25,train: epoch:  1, epoch_time:4392.189, batch_id:680/868, batch_time:2.173, loss is:0.000, train_accuracy:0.973
    time:2022-12-01 21:02:32,train: epoch:  1, epoch_time:4519.402, batch_id:700/868, batch_time:2.156, loss is:0.000, train_accuracy:0.974
    time:2022-12-01 21:04:38,train: epoch:  1, epoch_time:4645.919, batch_id:720/868, batch_time:2.211, loss is:0.000, train_accuracy:0.974
    time:2022-12-01 21:06:48,train: epoch:  1, epoch_time:4776.125, batch_id:740/868, batch_time:2.357, loss is:0.000, train_accuracy:0.975
    time:2022-12-01 21:08:59,train: epoch:  1, epoch_time:4906.449, batch_id:760/868, batch_time:2.301, loss is:0.000, train_accuracy:0.974
    time:2022-12-01 21:11:08,train: epoch:  1, epoch_time:5036.152, batch_id:780/868, batch_time:2.257, loss is:0.001, train_accuracy:0.975
    time:2022-12-01 21:13:18,train: epoch:  1, epoch_time:5165.823, batch_id:800/868, batch_time:2.235, loss is:0.169, train_accuracy:0.976
    time:2022-12-01 21:15:27,train: epoch:  1, epoch_time:5294.370, batch_id:820/868, batch_time:2.270, loss is:0.000, train_accuracy:0.976
    time:2022-12-01 21:17:33,train: epoch:  1, epoch_time:5421.023, batch_id:840/868, batch_time:2.209, loss is:0.000, train_accuracy:0.977
    time:2022-12-01 21:19:40,train: epoch:  1, epoch_time:5548.026, batch_id:860/868, batch_time:2.328, loss is:0.000, train_accuracy:0.977
    time:2022-12-01 21:20:34,train: epoch:  2, epoch_time: 2.187, batch_id: 0/868, batch_time:2.185, loss is:0.000, train_accuracy:1.000
    time:2022-12-01 21:22:45,train: epoch:  2, epoch_time:133.121, batch_id:20/868, batch_time:2.237, loss is:0.004, train_accuracy:1.000
    time:2022-12-01 21:24:55,train: epoch:  2, epoch_time:263.337, batch_id:40/868, batch_time:2.143, loss is:0.000, train_accuracy:1.000
    time:2022-12-01 21:27:05,train: epoch:  2, epoch_time:392.472, batch_id:60/868, batch_time:2.310, loss is:0.000, train_accuracy:1.000
    time:2022-12-01 21:29:10,train: epoch:  2, epoch_time:517.718, batch_id:80/868, batch_time:2.113, loss is:0.000, train_accuracy:1.000
    time:2022-12-01 21:31:17,train: epoch:  2, epoch_time:644.986, batch_id:100/868, batch_time:2.182, loss is:0.000, train_accuracy:1.000
    ^C


# 四、预测
## 4.1、将test.csv转为模型所需的npy格式


```python
# 将test.csv转为npy
test_csv_path = f'{parent_path}/test.csv'
# 度csv数据
value_list = []
with open(test_csv_path, encoding='utf-8') as csf:
    read_train_csv = csv.reader(csf)
    head = next(read_train_csv)
    for row in read_train_csv:
        value_float = [float(x) for x in row]
        value_list.append(value_float)
# 保存
save_np = np.array(value_list)
np.save(test_csv_path[:-4] + '.npy', save_np)
```

## 4.2、预测并压缩结果文件为提交格式


```python
!python predict.py \
        --model_name NeuralNetworkStructure \
        --save_train_best_model save_model/best96.6/train_60.pdparams \
        --BATCH_SIZE 2
!zip submission60.zip submission.csv
!rm -rf submission.csv
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
                        model_name│ NeuralNetworkStructure
    ──────────────────────────────┼──────────────────────────────────────────────────
                          model_id│ -1
    ──────────────────────────────┼──────────────────────────────────────────────────
              save_train_model_dir│ save_model/train
    ──────────────────────────────┼──────────────────────────────────────────────────
             save_train_best_model│ save_model/best96.6/train_60.pdparams
    ──────────────────────────────┼──────────────────────────────────────────────────
              save_infer_model_dir│ save_model/infer
    ──────────────────────────────┼──────────────────────────────────────────────────
                               gpu│ True
    ──────────────────────────────┼──────────────────────────────────────────────────
                          rgb_mean│ (67.0, 33.0, 11.0)
    ──────────────────────────────┼──────────────────────────────────────────────────
                           rgb_std│ (75.0, 40.0, 19.0)
    ──────────────────────────────┼──────────────────────────────────────────────────
                             aug_p│ 0.3
    ──────────────────────────────┼──────────────────────────────────────────────────
                       num_classes│ 6
    ──────────────────────────────┼──────────────────────────────────────────────────
                        label_text│ /ing/GAMMA.txt
    ──────────────────────────────┼──────────────────────────────────────────────────
                       train_label│ /yhdfev_jpg_one/voc_train.txt
    ──────────────────────────────┼──────────────────────────────────────────────────
                         train_dir│ /sdbhdfev_jpg_one
    ──────────────────────────────┼──────────────────────────────────────────────────
                         dev_label│ /sdb1dfhv_jpg_one/voc_train.txt
    ──────────────────────────────┼──────────────────────────────────────────────────
                           dev_dir│ /sdb1dfhdev_jpg_one
    ──────────────────────────────┼──────────────────────────────────────────────────
                          test_dir│ /homedhdt_jpg_one
    ──────────────────────────────┼──────────────────────────────────────────────────
                        INPUT_SIZE│ (64, 64)
    ──────────────────────────────┼──────────────────────────────────────────────────
                        BATCH_SIZE│ 2
    ──────────────────────────────┼──────────────────────────────────────────────────
                                lr│ 0.001
    ──────────────────────────────┼──────────────────────────────────────────────────
                        label_dict│ {'LAYING': 0, 'STANDING': 1, 'SITTING': 2, 'WALKING': 3, 'WALKING_UPSTAIRS': 4, 'WALKING_DOWNSTAIRS': 5}
    ──────────────────────────────┼──────────────────────────────────────────────────
    WARNING: Detect dataset only contains single fileds, return format changed since Paddle 2.1. In Paddle <= 2.0, DataLoader add a list surround output data(e.g. return [data]), and in Paddle >= 2.1, DataLoader return the single filed directly (e.g. return data). For example, in following code: 
    
    import numpy as np
    from paddle.io import DataLoader, Dataset
    
    class RandomDataset(Dataset):
        def __getitem__(self, idx):
            data = np.random.random((2, 3)).astype('float32')
    
            return data
    
        def __len__(self):
            return 10
    
    dataset = RandomDataset()
    loader = DataLoader(dataset, batch_size=1)
    data = next(loader())
    
    In Paddle <= 2.0, data is in format '[Tensor(shape=(1, 2, 3), dtype=float32)]', and in Paddle >= 2.1, data is in format 'Tensor(shape=(1, 2, 3), dtype=float32)'
    
     2000/2000updating: submission.csv (deflated 94%)


请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

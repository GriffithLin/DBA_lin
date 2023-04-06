#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/26 8:54
# @Author : fhh
# @FileName: main.py
# @Software: PyCharm

import os
import csv
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from train import DataTrain, predict, CosineScheduler
from config import get_config
from models.model import TextCNN
from models.tc_cbam import TC_CBAM

import estimate

torch.manual_seed(20230226)  # 固定随机种子
torch.backends.cudnn.deterministic = True  # 固定GPU运算方式
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
print(DEVICE)

amino_acids = 'XATCG'


def mark_label(src):
    assert src.find("hgmd")>=0 or src.find("pos")>=0 or src.find("gnomAD")>=0 or src.find("neg")>=0 or src.find("Neg")>=0 or src.find("Pos")>=0

    if src.find("hgmd")>=0 or src.find("pos")>=0 or src.find("Pos")>=0:
        return 1
    return 0

def getSequenceData(direction):
    # 从目标路径加载数据

    data_Frame = pd.read_csv(direction)
    data = data_Frame["before_mutation"] + data_Frame["after_mutation"]
    label = data_Frame.apply(lambda x: mark_label(x["src"]), axis = 1)
    return np.array(data), np.array(label)

def getSequenceData_Kmer(direction, k_mer):
    data_Frame = pd.read_csv(direction)
    data1 = data_Frame[str(k_mer)+"mer_before"]
    data2 =data_Frame[str(k_mer) + "mer_after"]
    label = data_Frame.apply(lambda x: mark_label(x["src"]), axis=1)
    return np.array(data1),np.array(data2), np.array(label)

def PadEncode(data, label, max_len=50):
    # 序列编码
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        if len(data[i]) > max_len:  # 剔除序列长度大于max_len的序列
            continue
        element, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:  # 剔除包含非天然氨基酸的序列
                sign = 1
                break
            index = amino_acids.index(j)
            element.append(index)
            sign = 0

        if length <= max_len and sign == 0:  # 序列长度小于50且只包含天然氨基酸的序列
            temp.append(element)
            seq_length.append(len(temp[b]))  # 保存序列有效长度
            b += 1
            element += [0] * (max_len - length)  # 用0补齐序列长度
            data_e.append(element)
            label_e.append(label[i])
    return torch.LongTensor(np.array(data_e)), torch.LongTensor(np.array(label_e))

def PadEncode_kmer(data1, data2, label, max_len=50):
    # 序列编码
    data1_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data1)):
        length1 = len(data1[i])
        length2 = len(data2[i])
        if max(len(data1[i]), len(data2[i])) > max_len:  # 剔除序列长度大于max_len的序列
            continue
        element1 = data1[i]
        element2 = data2[i]

        element1 += [0] * (max_len - length1)  # 用0补齐序列长度
        element2 += [0] * (max_len - length2)

        data_e.append(element1 + element2)
        label_e.append(label[i])

    return torch.LongTensor(np.array(data_e)), torch.LongTensor(np.array(label_e))


def data_load(train_direction, test_direction, max_length, batch, k_mer, encode='embedding'):
    assert encode in ['embedding', 'sequence'], 'There is no such representation!!!'
    # 从目标路径加载数据
    if k_mer == 0:
        train_sequence_data, train_sequence_label = getSequenceData(train_direction)
        test_sequence_data, test_sequence_label = getSequenceData(test_direction)

        # 选择序列编码方式
        # The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
        x_train, y_train= PadEncode(train_sequence_data, train_sequence_label, max_length)
        x_test, y_test= PadEncode(test_sequence_data, test_sequence_label, max_length)
    # else:
    #     train_sequence_data_before, train_sequence_data_after, train_sequence_label = getSequenceData_Kmer(train_direction, k_mer)
    #     test_sequence_data_before, test_sequence_data_after, test_sequence_label = getSequenceData_Kmer(test_direction, k_mer)
    #
    #     # 选择序列编码方式
    #     # The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
    #     x_train, y_train= PadEncode_kmer(train_sequence_data_before, train_sequence_data_after, train_sequence_label, max_length)
    #     x_test, y_test= PadEncode_kmer(test_sequence_data_before, test_sequence_data_after, test_sequence_label, max_length)




    # Create datasets
    dataset_train = TensorDataset(x_train, y_train)
    dataset_test = TensorDataset(x_test, y_test)
    dataset_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=True)
    return dataset_train, dataset_test


def spent_time(start, end):
    epoch_time = end - start
    minute = int(epoch_time / 60)  # 分钟
    secs = int(epoch_time - minute * 60)  # 秒
    return minute, secs

# '%.3f' % test_score[5],
def save_results(model_name, start, end, test_score, file_path):
    # 保存模型结果 csv文件
    title = ['Model']
    title.extend(test_score.keys())
    title.extend(['RunTime', 'Test_Time'])

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    content_row = [model_name]
    for key in test_score:
        content_row.append('%.3f' % test_score[key])
    content_row.extend([[end - start], now])

    content = [content_row]

    if os.path.exists(file_path):
        data = pd.read_csv(file_path, header=None)
        one_line = list(data.iloc[0])
        if one_line == title:
            with open(file_path, 'a+', newline='') as t:  # newline用来控制空的行数
                writer = csv.writer(t)  # 创建一个csv的写入器
                writer.writerows(content)  # 写入数据
        else:
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(title)  # 写入标题
                writer.writerows(content)
    else:
        with open(file_path, 'a+', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(title)
            writer.writerows(content)

def print_score(test_score):
    if type(test_score) == dict:
        for key in test_score:
            print(key + f': {test_score[key]:.3f}')
    else:
        print(test_score)

def main(args, paths=None):
    start_time = time.time()
    file_path = "{}/{}.csv".format('result', 'test')  # 结果保存路径

    print("Data is loading......（￣︶￣）↗　")
    train_dataset, test_dataset = data_load(args.train_direction, args.test_direction, args.max_length,
                                            args.batch_size,args.k_mer, encode='embedding')  # 加载训练数据和测试数据，并编码
    print("Data is loaded!ヾ(≧▽≦*)o")

    all_test_score = 0  # 初始话评估指标
    # 训练并保存模型
    if paths is None:
        print(f"{args.model_name} is training......")

        for counter in range(args.model_num):
            train_start = time.time()
            # 初始化相关参数
            model = TextCNN(args.vocab_size, args.embedding_size, args.filter_num, args.filter_size,
                            args.output_size, args.dropout)  # 初始化模型

            print(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # 优化器
            lr_scheduler = CosineScheduler(10000, base_lr=args.learning_rate, warmup_steps=500)  # 退化学习率
            criterion = torch.nn.BCEWithLogitsLoss()  # 损失函数
            # 初始化训练类
            Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)

            # 训练模型
            Train.train_step(train_dataset, epochs=args.epochs, model_num=counter)

            # 保存模型
            PATH = os.getcwd()
            each_model = os.path.join(PATH, 'saved_models', args.model_name + str(counter) + '.pth')
            torch.save(model.state_dict(), each_model)

            # 模型预测
            model_predictions, true_labels = predict(model, test_dataset, device=DEVICE)

            # 模型评估
            test_score = estimate.evaluate(model_predictions, true_labels, args.threshold)

            # 保存评估结果
            train_end = time.time()
            save_results(parse.model_name, train_start, train_end, test_score, file_path)

            # 打印评估结果
            print(f"{args.model_name}:{counter + 1}")
            print("测试集：")
            print_score(test_score)
            df_test_score = pd.DataFrame(test_score, index=[0])
            if type(all_test_score) == int:
                all_test_score = df_test_score
            else:
                all_test_score = all_test_score + df_test_score
    else:
        # 测试模型
        for model_path in paths:
            EStart_Time = time.time()
            # 初始化模型
            model = TC_CBAM(args.vocab_size, args.embedding_size, args.filter_num, args.filter_size,
                            args.output_size, args.dropout)
            # 加载模型
            model.load_state_dict(torch.load(model_path))
            # 模型预测
            model_predictions, true_labels = predict(model, test_dataset, device=DEVICE)
            # 模型评估
            test_score = estimate.evaluate(model_predictions, true_labels, args.threshold)
            print(f"{args.model_name}:")
            print("测试集：")
            print_score(test_score)

            df_test_score = pd.DataFrame(test_score, index=[0])
            if type(all_test_score) == int:
                all_test_score = df_test_score
            else:
                all_test_score = all_test_score + df_test_score

            EEnd_Time = time.time()
            # 保存模型性能
            save_results(parse.model_name, EStart_Time, EEnd_Time, test_score, file_path)

    "-------------------------------------------打印平均结果-----------------------------------------------"
    run_time = time.time()
    m, s = spent_time(start_time, run_time)  # 运行时间
    print(f"runtime:{m}m{s}s")
    print("测试集：")
    all_test_score = all_test_score / args.model_num
    print_score(all_test_score)
    save_results('average', start_time, run_time, all_test_score, file_path)
    "---------------------------------------------------------------------------------------------------"


if __name__ == '__main__':
    parse = get_config()  # 获取参数
    parse.model_name = 'tc_cbam_dro0.1'
    # parse.model_num = 1
    # parse.threshold = 0.55
    path = []
    for num in range(parse.model_num):
        a = f'saved_models/tc_cbam{num}.pth'
        path.append(a)
    main(parse)

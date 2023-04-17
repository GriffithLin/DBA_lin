#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/27 15:00
# @Author : fhh
# @FileName: config.py
# @Software: PyCharm
import argparse


def get_config():
    parse = argparse.ArgumentParser(description='default config')
    # 数据参数
    parse.add_argument('-max_length', type=int, default=260,
                       help='Maximum length of peptide sequence')
    parse.add_argument('-vocab_size', type=int, default=4**6+1,
                       help='The size of the vocabulary')
    parse.add_argument('-output_size', type=int, default=1,
                       help='Number of peptide functions')

    parse.add_argument('-k_mer', type=int, default=0,
                       help='k of k-mer input data')

    parse.add_argument('-divide_validata', type=bool, default=False,
                       help='divide 20% traindata to validata ')

    # 训练参数
    parse.add_argument('-model_num', type=int, default=1,
                       help='Number of primary training models')
    parse.add_argument('-batch_size', type=int, default=64*4,
                       help='Batch size')
    parse.add_argument('-epochs', type=int, default=200)
    parse.add_argument('-learning_rate', type=float, default=0.0018)
    parse.add_argument('-threshold', type=float, default=0.5)
    parse.add_argument('-early_stop', type=int, default=10)

    # 模型参数
    parse.add_argument('-model_name', type=str, default='TextCNN',
                       help='Name of the model')
    parse.add_argument('-embedding_size', type=int, default=64*4,
                       help='Dimension of the embedding')
    parse.add_argument('-dropout', type=float, default=0.6)
    parse.add_argument('-filter_num', type=int, default=64*2,
                       help='Number of the filter')
    parse.add_argument('-filter_size', type=list, default=[2, 3, 4, 5],
                       help='Size of the filter')

    # 路径参数
    parse.add_argument('-train_direction', type=str, default='dataset/train.csv',
                       help='Path of the training data')
    parse.add_argument('-test_direction', type=str, default='dataset/test.csv',
                       help='Path of the test data')

    config = parse.parse_args()
    return config

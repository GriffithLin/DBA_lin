#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 10:28
# @Author  : ywh
# @File    : estimate.py
# @Software: PyCharm
# import
from sklearn.metrics import matthews_corrcoef,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve,auc, accuracy_score


# def auc(score_label, Y_True, threshold=0.55):


# TP FP TN FN Total (%) Sens Spec LR+ LR− PPV NPV AUC MCC

def evaluate(score_label, Y_True, threshold=0.55):

    evl_result = dict()
    # 将预测概率分数转为标签
    for i in range(len(score_label)):

        if score_label[i] < threshold:  # threshold
            score_label[i] = 0
        else:
            score_label[i] = 1
    y_hat = score_label

    acc = accuracy_score(Y_True, y_hat)

    tn, fp, fn, tp = confusion_matrix(Y_True, y_hat).ravel()
    evl_result = {"tp":tp, "fp":fp, "tn":tn,"fn":fn , "total":tn + fp + fn + tp }
    precision, recall, thresholds = precision_recall_curve(Y_True, score_label)
    # print("Matthews相关系数: "+str(matthews_corrcoef(Y_True,y_hat)))
    # print('sensitivity/recall:',tp/(tp+fn))
    # print('specificity:',tn/(tn+fp))
    # print("F1值: "+str(f1_score(Y_True,y_hat)))
    # print('false positive rate:',fp/(tn+fp))
    # print('false discovery rate:',fp/(tp+fp))
    # print('TN:',tn,'FP:',fp,'FN:',fn,'TP:',tp)

    Sens = tp / (tp + fn)
    PPV = tp/(tp + fp)
    NPV = tn/(tn + fn)
    auc_precision_recall = auc(recall, precision)
    mcc = matthews_corrcoef(Y_True, y_hat)

    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    F1 = f1_score(Y_True, y_hat)
    false_positive_rate = fp / (tn + fp)
    false_discovery_rate= fp / (tp + fp)

    evl_result.update({"Sens":Sens, "Spec": specificity, "PPV":PPV, "NPV":NPV, "AUC":auc_precision_recall, "MCC":mcc, "acc":acc})

    return evl_result
"""
@Time : 2021/3/16 15:33
"""
import xlrd
import random
import numpy as np
from keras.models import load_model
import sklearn
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, accuracy_score,roc_curve,auc
from sklearn import metrics
#import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import random
import xlrd

# from torch.utils.data import DataLoader
from rice import rice_data
INPUT_DIM = 4
TIME_STEPS = 71


# def read_seq_label(filename):
#     workbook = xlrd.open_workbook(filename=filename)
#
#     booksheet_train = workbook.sheet_by_index(0)
#     nrows_train = booksheet_train.nrows
#
#     booksheet_val = workbook.sheet_by_index(1)
#     nrows_val = booksheet_val.nrows
#
#     booksheet_test = workbook.sheet_by_index(2)
#     nrows_test = booksheet_test.nrows
#     print(nrows_test)
#     seq_train=[]
#     seq_val=[]
#     seq_test = []
#     label_train = []
#     label_val=[]
#     label_test=[]
#     for i in range(nrows_train):
#         seq_train.append(booksheet_train.row_values(i)[0])
#         label_train.append(booksheet_train.row_values(i)[1])
#     for j in range(nrows_val):
#         seq_val.append((booksheet_val.row_values(j)[0]))
#         label_val.append(booksheet_val.row_values(j)[1])
#     for k in range(nrows_test):
#         seq_test.append((booksheet_test.row_values(k)[0]))
#         label_test.append(booksheet_test.row_values(k)[1])
#
#     return seq_train, np.array(label_train).astype(int),seq_val, np.array(label_val).astype(int),seq_test, np.array(label_test).astype(int)

def read_seq_label(filename):
    workbook = xlrd.open_workbook(filename=filename)



    booksheet_test = workbook.sheet_by_index(2)
    nrows_test = booksheet_test.nrows

    seq_test = []
    label_test=[]

    for k in range(nrows_test):
        seq_test.append((booksheet_test.row_values(k)[0]))
        label_test.append(booksheet_test.row_values(k)[1])

    return seq_test, np.array(label_test).astype(int)

def seq_to01_to0123(seq):


    nrows = len(seq)
    seq_len = len(seq[0])

    seq_01 = np.zeros((nrows, seq_len, 4), dtype='int')
    seq_0123 = np.zeros((nrows, seq_len), dtype='int')

    for i in range(nrows):
        one_seq = seq[i]
        one_seq = one_seq.replace('A', '0')
        one_seq = one_seq.replace('C', '1')
        one_seq = one_seq.replace('G', '2')
        one_seq = one_seq.replace('T', '3')
        seq_start = 0
        for j in range(seq_len):
            seq_0123[i, j] = int(one_seq[j - seq_start])
            if j < seq_start:
                seq_01[i, j, :] = 0.25
            else:
                try:
                    seq_01[i, j, int(one_seq[j - seq_start])] = 1
                except:
                    seq_01[i, j, :] = 0.25

    return seq_01

def get_fea(filename):

    seq_test,label_test= read_seq_label(filename)
    # x_train=seq_to01_to0123(seq_train)
    # x_val=seq_to01_to0123(seq_val)
    x_test=seq_to01_to0123(seq_test)
    # train_data = []
    # val_data = []
    test_data = []
    # y_train = np.vstack(label_train)
    # y_val = np.vstack(label_val)
    y_test = np.vstack(label_test)
    # x_train=x_train.astype(float)
    # x_val = x_val.astype(float)

    #norm = nn.LayerNorm([2, 3], elementwise_affine=True)
    #data = norm(out_list)
    #data.state_dict()


    # for i in range(y_train.shape[0]):
    #     train_data.append((x_train[i], y_train[i]))
    # for i in range(y_val.shape[0]):
    #     val_data.append((x_val[i], y_val[i]))
    for i in range(y_test.shape[0]):
        test_data.append((x_test[i], y_test[i]))

    #train_data = torch.Tensor(train_data)
    #val_data = torch.Tensor(val_data)
    #test_data = torch.Tensor(test_data)
    return x_test,y_test


def evaluate(predict_proba, predict_class, Y_test_array):

    acc = accuracy_score(Y_test_array, predict_class)
    binary_acc = metrics.accuracy_score(Y_test_array, predict_class)
    precision = metrics.precision_score(Y_test_array, predict_class)
    recall = metrics.recall_score(Y_test_array, predict_class)
    f1 = metrics.f1_score(Y_test_array, predict_class)
    auc = metrics.roc_auc_score(Y_test_array, predict_proba)
    mcc = metrics.matthews_corrcoef(Y_test_array, predict_class)
    TN, FP, FN, TP = metrics.confusion_matrix(Y_test_array, predict_class).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)
    print("acc:",acc,"bin_acc",binary_acc,"precisoon",precision,"recall",recall,"f1",f1,"auc",auc,"mcc",mcc,"Sen",sensitivity,"Spe",specificity)



if __name__ == '__main__':

    filename ='D:\论文精读\Co6mA\dataset\melanogaster.xlsx'
    #[x_test, y_test] = get_fea(filename)
    pos = 'D:\论文精读\Co6mA\data/rice\shuffle_pos.txt'
    neg = 'D:\论文精读\Co6mA\data/rice\shuffle_neg.txt'
    x_train, x_valid, x_val1, x_test, y_train, y_valid, y_val1, y_test = rice_data(pos, neg)
    filepath1 = 'D:/论文精读\Co6mA\model\AL6mA\ALbestmodelrice.h5'
    #filepath2 = "D:\\论文精读\\4\\add\\model\\me\\logs_al_notSINGLE\\bestmodel.h5"
    # filepath = "model/th/logs_al_notSINGLE/bestmodel.h5"
    # filepath = "model/me/logs_la_notSINGLE/bestmodel.h5"
    # filepath = "model/me/logs_al_notSINGLE/bestmodel.h5"
    m1 = load_model(filepath1,compile=False)
    #m2 = load_model(filepath2, compile=False)
    predict1 = m1.predict(x_test)
    auc = roc_auc_score(y_test, predict1)

    predicty1 = []
       #将数组或矩阵转换成列表
    #predict2 = m2.predict(x_test)
    #y_test = y_test.tolist()
    for indexResults in range(len(predict1)):
        if float(predict1[indexResults]) > 0.5 or float(
                predict1[indexResults]) == 0.5:
            predicty1.append(1)
        else:
            predicty1.append(0)
    evaluate(predict1, predicty1, y_test)


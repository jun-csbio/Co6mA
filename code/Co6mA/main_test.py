"""
@Time : 2021/3/16 15:33
"""
import xlrd
import random
import os
import numpy as np
from tensorflow.keras.models import load_model
import sklearn
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, accuracy_score,roc_curve,auc,precision_recall_curve
from sklearn import metrics
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from rice import rice_data
np.set_printoptions(threshold=np.inf)
import numpy as np
import tensorflow as tf
import random
import xlrd
import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader

INPUT_DIM = 4
TIME_STEPS = 71


def read_seq_label(filename):
    workbook = xlrd.open_workbook(filename=filename)



    booksheet_test = workbook.sheet_by_index(3)
    nrows_test = booksheet_test.nrows
    print(nrows_test)

    seq_test = []
    label_test=[]

    for k in range(nrows_test):
        seq_test.append((booksheet_test.row_values(k)[0]))
        label_test.append(booksheet_test.row_values(k)[1])

    return seq_test, np.array(label_test).astype(int)


def seq_to01_to0123(seq):


    nrows = len(seq)
    seq_len = len(seq[0])

    seq_01 = np.zeros((nrows, seq_len, 4), dtype='float32')
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
    x_test=seq_to01_to0123(seq_test)
    test_data = []
    y_test = np.vstack(label_test)

    return x_test,y_test

def evaluate(predict_proba, predict_class, Y_test_array):

    acc = accuracy_score(Y_test_array, predict_class)
    binary_acc = metrics.accuracy_score(Y_test_array, predict_class)
    precision = metrics.precision_score(Y_test_array, predict_class)
    recall = metrics.recall_score(Y_test_array, predict_class)
    f1 = metrics.f1_score(Y_test_array, predict_class)
    #auc = metrics.roc_auc_score(Y_test_array, predict_proba)
    mcc = metrics.matthews_corrcoef(Y_test_array, predict_class)
    TN, FP, FN, TP = metrics.confusion_matrix(Y_test_array, predict_class).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)
    print("acc:",acc,"bin_acc",binary_acc,"precision",precision,"recall",recall,"f1",f1,"mcc",mcc,"sen",sensitivity,"spe",specificity)
    return acc,precision,recall,f1,mcc,sensitivity,specificity

if __name__ == '__main__':
    #
    file_name= 'D:\论文精读\Co6mA\dataset\melanogaster.xlsx'
    x_test, y_test = get_fea(file_name)
    # pos = 'D:\论文精读\Co6mA\data/rice\shuffle_pos.txt'
    # neg = 'D:\论文精读\Co6mA\data/rice\shuffle_neg.txt'
    #x_train, x_val, x_val1, x_test, y_train, y_val, y_val1, y_test = rice_data(pos, neg)
    # pos = 'D:\课题/6mA/new_paper\TC-6mA-Pred-main\Datasets/pos_train.txt'
    # neg = 'D:\课题/6mA/new_paper\TC-6mA-Pred-main\Datasets/neg_train.txt'
    # pos_test = 'D:\课题/6mA/new_paper\TC-6mA-Pred-main\Datasets/pos.txt'
    # neg_test = 'D:\课题/6mA/new_paper\TC-6mA-Pred-main\Datasets/neg.txt'
    # pos = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/6mA/6mA_A.thaliana/train_pos.txt'
    # neg = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/6mA/6mA_A.thaliana/train_neg.txt'
    # pos_test = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/6mA/6mA_A.thaliana/test_pos.txt'
    # neg_test = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/6mA/6mA_A.thaliana/test_neg.txt'
    # x_train, x_val,x_val1, x_test, y_train, y_val,y_val1, y_test = rice_data(pos, neg, pos_test, neg_test)
    filepath1 = 'D:\论文精读\Co6mA/model/CBi6mA/thebestmodelme.h5'
    filepath2 = 'D:\论文精读\Co6mA/model/LA6mA/LAbestmodelme.h5'


    m1 = load_model(filepath1,compile=False)
    m1.summary()
    m2=load_model(filepath2,compile=False)
    m2.summary()
    # m3=load_model(filepath3,compile=False)
    # m3.summary()
    # m4=load_model(filepath4,compile=False)
    # m4.summary()


    predict1 = m1.predict(x_test,batch_size=512)
    predict2= m2.predict(x_test,batch_size=512)
    # predict3 = m3.predict(x_test, batch_size=512)
    # predict4 = m4.predict(x_test, batch_size=512)

    x = np.hstack((predict1,predict2))
    predict5 = np.average(np.mat(x) ,axis=1,weights=[0.56,1-0.56])
    predicty = []

    for indexResults in range(len(predict5)):
        if float(predict5[indexResults]) > 0.5 or float(
                predict5[indexResults]) == 0.5:
            predicty.append(1)
        else:
            predicty.append(0)
    acc,precision,recall,f1,mcc,sensitivity,specificity = evaluate(predict5, predicty, y_test)
    auc1 = roc_auc_score(y_test, predict5)
    print(auc1)

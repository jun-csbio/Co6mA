"""
@Time : 2021/3/16 15:15
"""
import os

import tensorflow
import xlrd
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from MyModel import LA6mA, AL6mA,LA6mA_al, AL6mA_al
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, accuracy_score
from sklearn import metrics

APPLY_ATTENTION_BEFORE_LSTM = False
# APPLY_ATTENTION_BEFORE_LSTM = True
INPUT_DIM = 4
TIME_STEPS = 41

import numpy as np
import tensorflow as tf
import random
import xlrd
from rice import rice_data
# from torch.utils.data import DataLoader

INPUT_DIM = 4
TIME_STEPS = 71

np.random.seed(1337)

def read_seq_label(filename):
    workbook = xlrd.open_workbook(filename=filename)

    booksheet_train = workbook.sheet_by_index(0)
    nrows_train = booksheet_train.nrows

    booksheet_val = workbook.sheet_by_index(1)
    nrows_val = booksheet_val.nrows

    booksheet_test = workbook.sheet_by_index(2)
    nrows_test = booksheet_test.nrows
    print(nrows_test)
    seq_train=[]
    seq_val=[]
    seq_test = []
    label_train = []
    label_val=[]
    label_test=[]
    for i in range(nrows_train):
        seq_train.append(booksheet_train.row_values(i)[0])
        label_train.append(booksheet_train.row_values(i)[1])
    for j in range(nrows_val):
        seq_val.append((booksheet_val.row_values(j)[0]))
        label_val.append(booksheet_val.row_values(j)[1])
    for k in range(nrows_test):
        seq_test.append((booksheet_test.row_values(k)[0]))
        label_test.append(booksheet_test.row_values(k)[1])

    return seq_train, np.array(label_train).astype(int),seq_val, np.array(label_val).astype(int),seq_test, np.array(label_test).astype(int)

def seq_to01_to0123(seq):


    nrows = len(seq)
    seq_len = len(seq[0])

    seq_01 = np.zeros((nrows, seq_len, 4), dtype='int')
    seq_0123 = np.zeros((nrows, seq_len), dtype='int')

    for i in range(nrows):
        one_seq = seq[i]
        if 'N' in one_seq:
            one_seq = str(one_seq).replace('N', 'A')
        one_seq = str(one_seq).replace('A', '0')
        one_seq = str(one_seq).replace('C', '1')
        one_seq = str(one_seq).replace('G', '2')
        one_seq = str(one_seq).replace('T', '3')
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
    seq_train,label_train,seq_val,label_val,seq_test,label_test= read_seq_label(filename)
    x_train=seq_to01_to0123(seq_train)
    x_val=seq_to01_to0123(seq_val)
    x_test=seq_to01_to0123(seq_test)
    print(len(x_train),len(x_val),len(x_test))
    train_data = []
    val_data = []
    test_data = []
    y_train = np.vstack(label_train)
    y_val = np.vstack(label_val)
    y_test = np.vstack(label_test)
    # x_train=x_train.astype(float)
    # x_val = x_val.astype(float)

    #norm = nn.LayerNorm([2, 3], elementwise_affine=True)
    #data = norm(out_list)
    #data.state_dict()


    for i in range(y_train.shape[0]):
        train_data.append((x_train[i], y_train[i]))
    for i in range(y_val.shape[0]):
        val_data.append((x_val[i], y_val[i]))
    for i in range(y_test.shape[0]):
        test_data.append((x_test[i], y_test[i]))

    #train_data = torch.Tensor(train_data)
    #val_data = torch.Tensor(val_data)
    #test_data = torch.Tensor(test_data)
    return x_train, y_train, x_val, y_val,x_test,y_test


if __name__ == '__main__':
    pos = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/6mA/train_pos.txt'
    neg = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/6mA/train_neg.txt'
    pos_test = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/6mA/test_pos.txt'
    neg_test = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/6mA/test_neg.txt'
    x_train, x_val, x_val1, x_test, y_train, y_val, y_val1, y_test = rice_data(pos, neg, pos_test, neg_test)
    # pos = 'D:\论文精读\Co6mA\data/rice\shuffle_pos.txt'
    # neg = 'D:\论文精读\Co6mA\data/rice\shuffle_neg.txt'
    #x_train, x_valid, x_val1, x_test, y_train, y_valid, y_val1, y_test = rice_data(pos, neg)
    #[x_train, y_train, x_valid, y_valid, _, _] = get_fea(filename)
    # pos = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/TC/pos_train.txt'
    # neg = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/TC/neg_train.txt'
    # pos_test = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/TC/pos.txt'
    # neg_test = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/TC/neg.txt'
    #x_train, x_val,x_val1, x_test, y_train, y_val,y_val1, y_test = rice_data(pos, neg, pos_test, neg_test)
    # y_train = np.vstack(y_train)     #按行方向堆积成一个新的数组
    # y_valid = np.vstack(y_valid)


    # if APPLY_ATTENTION_BEFORE_LSTM:
    m =  LA6mA()
    #else:
        #m = LA6mA()

    m.summary()

    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #bestpath = 'D:/论文精读\Co6mA/training/bestmodelme.h5'
    bestpath='/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/thebestmodelLA_6mA_A.thaliana.h5'

    #logging = tensorflow.keras.callbacks.TensorBoard(log_dir='/al/th/')    #保存TensorBoard要解析的日志文件的目录路径

    checkpoint = ModelCheckpoint(filepath=bestpath, monitor='val_loss', save_best_only=True,period=1)
    #mode = 'min'
        #ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)

    m.fit([x_train], y_train,
          batch_size=64,
          epochs=50,
          verbose=2,   #为每个epoch输出一行记录
          validation_data=([x_val], y_val),
          callbacks=[checkpoint, reduce_lr,early_stopping])   #回调函数




import numpy as np
import tensorflow as tf
import random
import xlrd

# from torch.utils.data import DataLoader

INPUT_DIM = 4
TIME_STEPS = 71


def read_seq_label(filename):
    workbook = xlrd.open_workbook(filename=filename)

    booksheet_train = workbook.sheet_by_index(0)
    nrows_train = booksheet_train.nrows

    booksheet_val = workbook.sheet_by_index(1)
    nrows_val = booksheet_val.nrows

    booksheet_val1 = workbook.sheet_by_index(2)
    nrows_val1 = booksheet_val1.nrows

    booksheet_test = workbook.sheet_by_index(3)
    nrows_test = booksheet_test.nrows
    print(nrows_test)
    seq_train=[]
    seq_val=[]
    seq_val1 = []
    seq_test = []
    label_train = []
    label_val=[]
    label_val1 = []
    label_test=[]
    for i in range(nrows_train):
        seq_train.append(booksheet_train.row_values(i)[0])
        label_train.append(booksheet_train.row_values(i)[1])
    for j in range(nrows_val):
        seq_val.append((booksheet_val.row_values(j)[0]))
        label_val.append(booksheet_val.row_values(j)[1])
    for w in range(nrows_val1):
        seq_val1.append((booksheet_val1.row_values(w)[0]))
        label_val1.append(booksheet_val1.row_values(w)[1])
    for k in range(nrows_test):
        seq_test.append((booksheet_test.row_values(k)[0]))
        label_test.append(booksheet_test.row_values(k)[1])

    return seq_train, np.array(label_train).astype(int),seq_val, np.array(label_val).astype(int),seq_val1, np.array(label_val1).astype(int),seq_test, np.array(label_test).astype(int)


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

    seq_train,label_train,seq_val,label_val,seq_val1,label_val1,seq_test,label_test= read_seq_label(filename)
    x_train=seq_to01_to0123(seq_train)
    x_val=seq_to01_to0123(seq_val)
    x_val1 = seq_to01_to0123(seq_val1)
    x_test=seq_to01_to0123(seq_test)
    print(len(x_train),len(x_val),len(x_test))
    train_data = []
    val_data = []
    val1_data = []
    test_data = []
    y_train = np.vstack(label_train)
    y_val = np.vstack(label_val)
    y_val1 = np.vstack(label_val1)
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
    for i in range(y_val1.shape[0]):
        val1_data.append((x_val1[i], y_val1[i]))
    for i in range(y_test.shape[0]):
        test_data.append((x_test[i], y_test[i]))

    #train_data = torch.Tensor(train_data)
    #val_data = torch.Tensor(val_data)
    #test_data = torch.Tensor(test_data)
    return x_train,x_val,x_val1,x_test,y_train,y_val,y_val1,y_test

# def data(filename):
#     Sentences = []
#     train_data,val_data,test_data,y_train,y_val,y_test = load_data2(filename)
#
#
#
#     train_len=len(y_train)
#     val_len = len(y_val)
#     test_len = len(y_test)
#     dec_input1 = tf.ones([train_len,1])
#     dec_input2 =  tf.ones([val_len,1])
#     dec_input3 =  tf.ones([test_len,1])
#
#
#
#     return dec_input1,dec_input2,dec_input3
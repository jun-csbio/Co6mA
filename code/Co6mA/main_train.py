# !/usr/bin/python
# vim: set fileencoding=utf-8 :
import sys

from keras import Input

sys.setrecursionlimit(15000)
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, accuracy_score,roc_curve,auc
from sklearn import metrics

#from data import get_fea
np.set_printoptions(threshold=np.inf)
from keras.optimizers import Adam,Nadam, SGD
#from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.models import Input, Model
from keras.layers.core import Permute, Reshape, Dense, Lambda, K, RepeatVector, Flatten

from keras_self_attention import SeqSelfAttention
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l1,l2

from keras.models import load_model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from rice import rice_data
from keras.layers import Multiply
import os
from data import get_fea
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
start = time.time()
# your code

#************************************
# load data
#************************************
np.random.seed(1337) # for reproducibility
print( 'loading data')

file_name ='D:\论文精读\Co6mA\dataset\A.thaliana.xlsx'
# pos = 'D:\课题/6mA/new_paper\TC-6mA-Pred-main\Datasets\pos_train.txt'
# neg = 'D:\课题/6mA/new_paper\TC-6mA-Pred-main\Datasets/neg_train.txt'
# pos_test='D:\课题/6mA/new_paper\TC-6mA-Pred-main\Datasets\pos.txt'
# neg_test='D:\课题/6mA/new_paper\TC-6mA-Pred-main\Datasets/neg.txt'

# pos = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/6mA/train_pos.txt'
# neg = '/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/6mA/train_neg.txt'
# pos_test='/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/6mA/test_pos.txt'
# neg_test='/data0/junh/stu/yuxuantang/6ma/Co6mA/training/EpiTEAmDNA-v1-02/6mA/test_neg.txt'
#x_train,x_val,x_val1,x_test,y_train,y_val,y_val1,y_test=rice_data(pos,neg,pos_test,neg_test)
x_train,x_val,x_val1,x_test,y_train,y_val,y_val1,y_test = get_fea(file_name)
INPUT_DIM = 4
TIME_STEPS = 41



print('building model...............')

model = Sequential()
lstm_units=256

#
# #*************************
# 1 convolutional layer
#*************************
NUM_FILTER1 = 80

#model.add(Bidirectional(LSTM(lstm_units,input_shape=(41,4) ,return_sequences=True),merge_mode='sum'))
#256-40-4-0.001  256-80-4-0.001 256
model.add(Convolution1D(input_dim=4,
                        input_length=x_train.shape[1],
                        filters=NUM_FILTER1,
                        kernel_size=4,
                        border_mode="valid",
            activation="linear",
                        subsample_length=1,
            #W_regularizer = l2(0.01),
            init='he_normal',
            name = "cov1"))

model.add(LeakyReLU(alpha=.001))
model.add(LeakyReLU(alpha=.001))
model.add(LeakyReLU(alpha=.001))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling1D(pool_length=2, stride=2))
model.add(Dropout(0.2))




# #*******************************
# 2 convolutional layer
# *******************************

model.add(Convolution1D(filters=80,
                        kernel_size=2,
                        border_mode="valid",
            activation="linear",
                        subsample_length=1, init='he_normal',
            name = "cov2"))

model.add(LeakyReLU(alpha=.001))
model.add(LeakyReLU(alpha=.001))
model.add(LeakyReLU(alpha=.001))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling1D(pool_length=2, stride=2))
model.add(Dropout(0.5))

# # #*******************************
# # 3 convolutional layer
#*******************************

model.add(Convolution1D(filters=80,
                        kernel_size=4,
                        border_mode="valid",
                        activation="linear",
                        subsample_length=1, init='he_normal',
			name="cov3"))

model.add(LeakyReLU(alpha=.001))
model.add(LeakyReLU(alpha=.001))
model.add(LeakyReLU(alpha=.001))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling1D(pool_length=2, stride=2))
model.add(Dropout(0.5))
# # #



model.add(Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='sum'))
model.add(Flatten())

model.add(LeakyReLU(alpha=.001))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.5))
model.add(Dense(output_dim=100))
model.add(Dense(output_dim=1))
model.add(Activation('sigmoid'))


#***********************************
# model training
#
#
#***********************************

print('compiling and fitting model...........')
abspath = os.getcwd()
savepath='D:\论文精读\Co6mA/code/Co6mA/thebestmodelth.h5'

def evaluate(predict_proba, predict_class, Y_test_array):

    acc = accuracy_score(Y_test_array, predict_class)
    binary_acc = metrics.accuracy_score(Y_test_array, predict_class)
    precision = metrics.precision_score(Y_test_array, predict_class)
    recall = metrics.recall_score(Y_test_array, predict_class)
    f1 = metrics.f1_score(Y_test_array, predict_class)
    mcc = metrics.matthews_corrcoef(Y_test_array, predict_class)
    TN, FP, FN, TP = metrics.confusion_matrix(Y_test_array, predict_class).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)
    print("acc:",acc,"bin_acc",binary_acc,"precision",precision,"recall",recall,"f1",f1,"mcc",mcc,"sen",sensitivity,"spe",specificity)

sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
checkpointer = ModelCheckpoint(filepath = savepath, verbose=1, monitor='val_accuracy', save_best_only=True)
earlystopper = EarlyStopping(monitor='loss', patience=50, verbose=1)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

Hist = model.fit(x_train, y_train, batch_size=512, nb_epoch=500, shuffle=True, verbose=2,validation_data=(x_val,y_val),
	   callbacks=[checkpointer,earlystopper])
time_used = time.time()-start
print('运行时间：',time_used)
model.summary()

#
#
from matplotlib import pyplot as plt
#
# # #绘图函数d
def print_history(history):
#     # 绘制训练 & 验证的准确率值
#
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['accuracy'])
    max_idx1 = np.argmax(history.history['val_accuracy'])
    print('val_acc',max_idx1)
    max_idx2 = np.argmax(history.history['accuracy'])
    print('acc',max_idx2)
    plt.title('Model accuracy&loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'Val_acc'])
    plt.savefig('./acc_th1.png')
    plt.show()
# print(Hist.params)
print_history(Hist) #调用绘图函数
# ###############################################    show   ###############################################
def print_history_loss(history):
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    max_idx3 = np.argmin(history.history['val_loss'])
    print('val_loss',max_idx3)
    max_idx4 = np.argmin(history.history['loss'])
    print('loss',max_idx4)
    plt.legend(['Train_loss', 'Val_loss'])
    plt.xlabel('Epoch')
    plt.savefig('./loss_th1.png')
    plt.show()
# print(Hist.params)
print_history_loss(Hist) #调用绘图函数








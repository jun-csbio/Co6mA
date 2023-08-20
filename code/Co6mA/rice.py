import numpy as np
from sklearn.model_selection import train_test_split
def readseq(posfile, negfile):  # return the peptides from input peptide list file
    posdata = open(posfile, 'r')
    pos = []
    for l in posdata.readlines():
        if l[0] == '>':
            continue
        else:
            pos.append(l.strip('\t0\n'))
    posdata.close()
    negdata = open(negfile, 'r')
    neg = []
    for l in negdata.readlines():
        if l[0] == '>':
            continue
        else:
            neg.append(l.strip('\t0\n'))
    negdata.close()
    #print(pos)
    #print(neg)
    return pos, neg
def seq_to01_to0123(seq):


    nrows = len(seq)
    seq_len = len(seq[0])

    seq_01 = np.zeros((nrows, seq_len, 4), dtype='int')
    seq_0123 = np.zeros((nrows, seq_len), dtype='int')

    for i in range(nrows):
        one_seq = seq[i]
        if 'N' in one_seq:
            one_seq = one_seq.replace('N', 'A')
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

def rice_data(pos,neg,pos_test,neg_test):
    pos, neg = readseq(pos,neg)
    x_test_pos,x_test_neg=readseq(pos_test,neg_test)
    data=pos+neg
    x_test = x_test_pos+x_test_neg
    target = [1] * len(pos) + [0] * len(neg)
    x_trainval, x_val, y_trainval,y_val= train_test_split(data, target, random_state=42,test_size=1.0/8)
    x_train, x_val1, y_train, y_val1 = train_test_split(x_trainval, y_trainval, random_state=42, test_size=1.0/ 8)
    #x_val1, x_val2, y_val1, y_val2 = train_test_split(x_val, y_val, random_state=42, test_size=1.0 / 8)
    x_train = seq_to01_to0123(x_train)
    x_val1 = seq_to01_to0123(x_val1)
    x_val = seq_to01_to0123(x_val)
    x_test = seq_to01_to0123(x_test)
    y_test = [1] * len(x_test_pos) + [0] * len(x_test_neg)
    return x_train,x_val,x_val1,x_test,y_train,y_val,y_val1,y_test

# x_train,x_val,x_test,y_train,y_val,y_test =\
#     rice_data("D:\课题/6mA/new_paper\i6mA-vote-master\datasets\Training_Dataset\positive_training_dataset_for_Rosaceae.txt",
#               "D:\课题/6mA/new_paper\i6mA-vote-master\datasets\Training_Dataset/negative_training_dataset_for_Rosaceae.txt",
#               "D:\课题/6mA/new_paper\i6mA-vote-master\datasets\Test_Dataset\positive_test_dataset_for_Arabidopsis_Thaliana.txt",
#               "D:\课题/6mA/new_paper\i6mA-vote-master\datasets\Test_Dataset/negative_test_dataset_for_Arabidopsis_Thaliana.txt")
# print('1')
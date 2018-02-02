# coding: utf-8

import os
import numpy as np
import scipy.io as scio

def loadsubject_data():
    train_data = np.empty((40091, 100, 150))
    test_data = np.empty((16487, 100, 150))
    train_label = np.empty((40091, ), dtype='uint8')
    test_label = np.empty((16487, ), dtype='uint8')
    label = np.empty((56578, ), dtype='uint8')
    files = os.listdir('/data/Datasets/tujh/NTU_skeleton_main_lstm')
    num = len(files)


    train_num = 0
    test_num = 0

    for i in range(num):
        one_data = scio.loadmat('/data/Datasets/tujh/NTU_skeleton_main_lstm/' + files[i])
        one_data = one_data['skeleton']

        # get the label
        temp = files[i][17:20]
        if list(temp)[0] == 0 and list(temp)[1] == 0:
            label[i] = int(list(temp)[2]) - 1
        else:
            label[i] = int(list(temp)[1] + list(temp)[2]) - 1


        temp1 = files[i][9:12]
        subject_num = int(list(temp1)[0] + list(temp1)[1] + list(temp1)[2])

        if subject_num == 1 or subject_num == 2 or subject_num == 4 or subject_num == 5 or subject_num == 8 or subject_num == 9\
                or subject_num == 13 or subject_num == 14 or subject_num == 15 or subject_num == 16 or subject_num == 17 \
                or subject_num == 18 or subject_num == 19 or subject_num == 25 or subject_num == 27 or subject_num == 28 \
                or subject_num == 31 or subject_num == 34 or subject_num == 35 or subject_num == 38:
            train_data[train_num,:,:] = one_data
            train_label[train_num] = label[i]
            if train_num < 40091:
                train_num = train_num + 1
                if train_num%1000==0:
                    print('train_num = ', train_num)
        else:
            test_data[test_num,:,:] = one_data
            test_label[test_num] = label[i]
            if test_num < 16487:
                test_num = test_num + 1
                if test_num%1000==0:
                    print('test_num = ', test_num)

    return train_data, train_label, test_data, test_label


def loadview_data():
    train_data = np.empty((37646, 100, 150))
    test_data = np.empty((18932, 100, 150))
    train_label = np.empty((37646, ), dtype='uint8')
    test_label = np.empty((18932, ), dtype='uint8')
    label = np.empty((56578, ), dtype='uint8')
    files = os.listdir('/data/tujh/NTU_skeleton_main_lstm')
    num = len(files)

    train_num = 0
    test_num = 0

    for i in range(num):
        one_data = scio.loadmat('/data/tujh/NTU_skeleton_main_lstm/' + files[i])
        one_data = one_data['skeleton']

        # get the label
        temp = files[i][17:20]
        if list(temp)[0] == 0 and list(temp)[1] == 0:
            label[i] = int(list(temp)[2]) - 1
        else:
            label[i] = int(list(temp)[1] + list(temp)[2]) - 1


        temp1 = files[i][5:8]
        view_num = int(list(temp1)[0] + list(temp1)[1] + list(temp1)[2])


        if view_num == 2 or view_num == 3:
            train_data[train_num,:,:] = one_data
            train_label[train_num] = label[i]
            if train_num < 37646:
                train_num = train_num + 1
                if train_num%1000==0:
                    print('train_num = ', train_num)
        else:
            test_data[test_num,:,:] = one_data
            test_label[test_num] = label[i]
            if test_num < 18932:
                test_num = test_num + 1
                if test_num%1000==0:
                    print('test_num = ', test_num)

    return train_data, train_label, test_data, test_label


if __name__ == '__main__':
    train_data, train_label, test_data, test_label  = load_data()


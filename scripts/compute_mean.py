import os
import h5py
import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser(description='Person Identification Dataset Script')
parser.add_argument('--path', default='./data/idrad', type=str)
parser.add_argument('--feature', default='microdoppler', type=str)
args = parser.parse_args()

if __name__ == '__main__':


    mds = np.zeros((0, 256))
    n = 0
    S = 0.0
    m = 0.0
    min = np.zeros((80, 128))
    max = np.zeros((80, 128))
    max[:] = -1e9


    a = np.zeros((0, 256))
    for filename in glob.glob(os.path.join(args.path, 'train', '*.hdf5')):
        file = h5py.File(filename, 'r')
        data = file[args.feature][:]
        a = np.concatenate((a, data))
        # for i in range(len(data)):
        #     n = n + 1
        #     m_prev = m
        #     m = m + (data[i] - m) / n
        #     S = S + (data[i] - m) * (data[i] - m_prev)
        #     min = np.min(np.concatenate((min[np.newaxis], data[i][np.newaxis])), axis=0)
        #     max = np.max(np.concatenate((max[np.newaxis], data[i][np.newaxis])), axis=0)

        file.close()


    # print(a.shape)
    # a = np.median(a, axis=0)
    # np.save('microdoppler_median.npy', a)
    # exit()


    print("min:" + str(np.min(a)))
    # print("min:" + str(np.min(mds)))
    print("max:" + str(np.max(a)))
    # print("max:" + str(np.max(mds)))
    # print("mean:" + str(np.mean(mds)))
    print("mean:" + str(np.mean(a)))
    print("std:" + str(np.std(a)))

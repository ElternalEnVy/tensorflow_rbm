import struct
import pickle
import os.path
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(mode='train', path='./data'):
    """
    Load and return MNIST dataset.

    Returns
    -------
    data : (n_samples, 784) np.ndarray
        Data representing raw pixel intensities (in [0., 255.] range).
    target : (n_samples,) np.ndarray
        Labels vector (zero-based integers).
    """
    dirpath = os.path.join(path, 'mnist/')
    if mode == 'train':
        fname_data = os.path.join(dirpath, 'train-images-idx3-ubyte')
        fname_target = os.path.join(dirpath, 'train-labels-idx1-ubyte')
    elif mode == 'test':
        fname_data = os.path.join(dirpath, 't10k-images-idx3-ubyte')
        fname_target = os.path.join(dirpath, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("`mode` must be 'train' or 'test'")

    with open(fname_data, 'rb') as fdata:
        magic, n_samples, n_rows, n_cols = struct.unpack(">IIII", fdata.read(16))
        data = np.fromfile(fdata, dtype=np.uint8)
        data = data.reshape(n_samples, n_rows * n_cols)

    with open(fname_target, 'rb') as ftarget:
        magic, n_samples = struct.unpack(">II", ftarget.read(8))
        target = np.fromfile(ftarget, dtype=np.int8)

    return data.astype(float), target

def load_cifar10(mode='train', path='/data'):
    """
    Load and return CIFAR-10 dataset.

    Returns
    -------
    data : (n_samples, 3 * 32 * 32) np.ndarray
        Data representing raw pixel intensities (in [0., 255.] range).
    target : (n_samples,append(np.asarray(img).astype(float))
        Labels vector (zappend(np.asarray(img).astype(float))
    """
    dirpath = os.path.join(path, 'cifar-10-batches-py/')
    batch_size = 10000
    if mode == 'train':
        fnames = ['data_batch_{0}'.format(i) for i in range(1, 5 + 1)]
    elif mode == 'test':
        fnames = ['test_batch']
    else:
        raise ValueError("`mode` must be 'train' or 'test'")
    n_samples = batch_size * len(fnames)
    data = np.zeros(shape=(n_samples, 3 * 32 * 32), dtype=float)
    target = np.zeros(shape=(n_samples,), dtype=int)
    start = 0
    for fname in fnames:
        fname = os.path.join(dirpath, fname)
        with open(fname, 'rb') as fdata:
            _data = pickle.load(fdata)
            data[start:(start + batch_size)] = np.asarray(_data['data'])
            target[start:(start + batch_size)] = np.asarray(_data['labels'])
        start += 10000
    return data, target

def load_NORB(mode='train', path='./data'):
    path = os.path.join(path, 'NORB/')
    if mode == 'train':
        fid_images = open(path + 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat','rb')
        fid_labels = open(path + 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat','rb')
    
    elif mode == 'test':
        fid_images = open(path + 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat','rb')
        fid_labels = open(path + 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat','rb')

    for i in range(6):
        a = fid_images.read(4)    #header

    num_images = 24300*2
    images = np.zeros((num_images,96,96))

    for idx in range(num_images):
        temp = fid_images.read(96*96)
        images[idx,:,:] = np.fromstring(temp,'uint8').reshape(96,96).T 

    for i in range(5):
        a = fid_labels.read(4) #header

    labels = np.fromstring(fid_labels.read(num_images*np.dtype('int32').itemsize),'int32')
    labels = np.repeat(labels,2)

    perm = np.random.permutation(num_images)
    images = images[perm]
    labels = labels[perm]
    labels = labels.reshape(images.shape[0],1) == np.arange(5) # one hot

    return images,labels

def load_OCR_letters(mode='train', path='./data'):

    path = os.path.join(path, 'OCR_letters/')
    data_filepath = os.path.join(path, 'ocr_letters_' + mode)
    label_filepath = os.path.join(path, 'ocr_letters_' + mode + '_label')
    data = []
    label = []

    data_file = open(data_filepath, 'rt')
    reader = csv.reader(data_file, delimiter='\t')
    lines = list(reader)

    for line in lines:
        data_line = [int(i) for i in line]
        data.append(data_line)
        
    label_file = open(label_filepath, 'rt')
    reader = csv.reader(label_file, delimiter='\n')
    lines = list(reader)

    for line in lines:
        label.append(int(line[0]))


    data = np.array(data).astype(float)
    label = np.asarray(label)

    return data, label

def divide_OCR_letters(path='./data'):
    path = os.path.join(path, 'OCR_letters/')
    filepath = os.path.join(path, 'letter.data')
    file = open(filepath, 'rt')
    reader = csv.reader(file, delimiter='\t')
    lines = list(reader)
    data, target = [], []
    for line in lines:
        target.append(ord(line[1]) - ord('a'))
        img = [int(x) for x in line[6:134]]
        data.append(img)
    data = np.array(data)
    target = np.asarray(target)
    perm = [i for i in range(len(data))]

    train_file, test_file = [open(os.path.join(path, 'ocr_letters_' + ds ), 'w') for ds in ['train', 'test']]
    train_label, test_label = [open(os.path.join(path, 'ocr_letters_' + ds + '_label'), 'w') for ds in ['train', 'test']]

    import random
    random.seed(12345)
    random.shuffle(perm)
    line_id = 0
    train_valid_split = 42152
    for i in perm:
        s = data[i]
        ss = ''
        for k in s[:-1]:
            ss = ss + str(k) + '\t'
        ss = ss + str(s[-1])
        t = str(target[i])
        if line_id < train_valid_split:
            train_file.write(ss + '\n')
            train_label.write(t + '\n')
        else:
            test_file.write(ss + '\n')
            test_label.write(t + '\n')
        line_id += 1

    print('Done!')
        

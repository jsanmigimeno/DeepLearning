## Modification from original HardNet implementation in 
## https://raw.githubusercontent.com/DagnyT/hardnet/master/code/dataloaders/HPatchesDatasetCreator.py
## I need to clean it a little bit and modify some things, but it works

import os
import numpy as np
import cv2
import sys
import json
import keras
from keras import backend as K
from tqdm import tqdm
import glob
import random
from keras.preprocessing.image import ImageDataGenerator as IDG

splits = ['a', 'b', 'c', 'view', 'illum']
tps = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']




class DenoiseHPatches(keras.utils.Sequence):
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps

    def __init__(self, seqs, batch_size = 32):
        self.all_paths = []
        self.batch_size = batch_size
        self.dim = (32, 32)
        self.n_channels = 1
        self.sequences = {}
        self.sequences_n = {}
        
        # Augmentation params
        self.transform = False
        self.rotationRange = [-90, 90]
        self.zoomRange = [0.8, 1.2]
        self.verticalFlip = True
        self.horizontalFlip = True
        self.fill_mode='nearest'

        for base in tqdm(seqs):
            name = base.split('/')
            self.name = name[-1]
            self.base = base
            for t in self.itr:
                im_path = os.path.join(base, t + '.png')
                img_n = cv2.imread(im_path.replace('.png', '_noise.png'), 0)
                img = cv2.imread(im_path, 0)
                N = img.shape[0] / 32
                seq_im = np.array(np.split(img, N),
                                  dtype=np.uint8)
                seq_im_n = np.array(np.split(img_n, N),
                                    dtype=np.uint8)
                for i in range(int(N)):
                    path = os.path.join(base, t, str(i) + '.png')
                    self.all_paths.append(path)
                    self.sequences[path] = seq_im[i]
                    self.sequences_n[path] = seq_im_n[i]
        self.on_epoch_end()

    def get_images(self, index):
        # If transform enabled, get augementation params
        params = {}
        if self.transform:
            params['theta'] = np.random.uniform(self.rotationRange[0], self.rotationRange[1])
            params['zx'] = np.random.uniform(self.zoomRange[0], self.zoomRange[1])
            params['zy'] = np.random.uniform(self.zoomRange[0], self.zoomRange[1])
            
            if self.verticalFlip:
                params['flip_vertical'] = bool(np.random.randint(2))
            if self.horizontalFlip:
                params['flip_horizontal'] = bool(np.random.randint(2))

        def transform_img(img, transform, params):
            if transform:
                IDGgen = IDG()
                img = np.expand_dims(img, -1)
                img = IDGgen.apply_transform(img, params)
                img = np.squeeze(img)
            return img

        path = self.all_paths[index]
        img = self.sequences[path]
        img_n = self.sequences_n[path]
        img = transform_img(img, self.transform, params).astype(np.float32)
        img_n = transform_img(img_n, self.transform, params).astype(np.float32)
        return img, img_n

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.all_paths) / self.batch_size))

    def __getitem__(self, index):
        # Define clean and noisy batch shapes
        img_clean = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_noise = np.empty((self.batch_size,) + self.dim + (self.n_channels,))

        # for each image in the batch
        for i in range(self.batch_size):
            img, img_n = self.get_images(index*self.batch_size+i)    
            img_clean[i] = np.expand_dims(img, -1)
            img_noise[i] = np.expand_dims(img_n, -1)

        return img_noise, img_clean    

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        random.shuffle(self.all_paths)

class hpatches_sequence_folder:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self, base, noise=1):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        if noise:
            noise_path = '_noise'
        else:
            noise_path = ''
        for t in self.itr:
            im_path = os.path.join(base, t+noise_path+'.png')
            im = cv2.imread(im_path,0)
            self.N = im.shape[0]/32
            setattr(self, t, np.split(im, self.N))



def generate_triplets(labels, num_triplets, batch_size, data=None, model=None):
    if model is not None:
        topPer = 0.25
        num_triplets = int(num_triplets/topPer)

    def create_indices(_labels):
        inds = dict()
        for idx, ind in enumerate(_labels):
            if ind not in inds:
                inds[ind] = []
            inds[ind].append(idx)
        return inds

    triplets = []
    indices = create_indices(np.asarray(labels))
    unique_labels = np.unique(np.asarray(labels))
    n_classes = unique_labels.shape[0]
    # add only unique indices in batch
    already_idxs = set()
    
    for x in tqdm(range(num_triplets)):
        if len(already_idxs) >= batch_size:
            already_idxs = set()
        c1 = np.random.randint(0, n_classes)
        while c1 in already_idxs:
            c1 = np.random.randint(0, n_classes)
        already_idxs.add(c1)
        c2 = np.random.randint(0, n_classes)
        while c1 == c2:
            c2 = np.random.randint(0, n_classes)
        if len(indices[c1]) == 2:  # hack to speed up process
            n1, n2 = 0, 1
        else:
            n1 = np.random.randint(0, len(indices[c1]))
            n2 = np.random.randint(0, len(indices[c1]))
            while n1 == n2:
                n2 = np.random.randint(0, len(indices[c1]))
        n3 = np.random.randint(0, len(indices[c2]))
        triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])

    if model is not None:
        scores = []
        N = len(triplets)
        allImgsA = np.empty((N, 32, 32, 1))
        allImgsB = np.empty((N, 32, 32, 1))
        allImgsC = np.empty((N, 32, 32, 1))

        for i, triplet in enumerate(triplets):
            a, p, n = data[triplet[0]], data[triplet[1]], data[triplet[2]]
            allImgsA[i] = np.expand_dims(a, -1).astype(float)
            allImgsB[i] = np.expand_dims(p, -1).astype(float)
            allImgsC[i] = np.expand_dims(n, -1).astype(float)
        print("End load images, predicting")
        scores = np.empty(N)

        for batchId in range(ceil(N/batch_size)):
            low = batchId*batch_size
            if (batchId + 1)*batch_size > N:
                high = N
            else:
                high = (batchId+1)*batch_size
            scores[low:high] = np.squeeze(model.predict([allImgsA[low:high], allImgsB[low:high], allImgsC[low:high]]))
        
        input()
        print("Start Sort")
        idx = np.flip(np.argsort(scores, kind='quicksort'), axis=0)
        input()
        print("End Sort")
        worstIdx = idx[:int(topPer*len(idx))]

        triplets = np.array(triplets)[worstIdx]
    else:
        triplets = np.array(triplets)

    return triplets


class HPatches():
    def __init__(self, train=True, transform=None, download=False, train_fnames=[],
                 test_fnames=[], denoise_model=None, use_clean=False):
        self.train = train
        self.transform = transform
        self.train_fnames = train_fnames
        self.test_fnames = test_fnames
        self.denoise_model = denoise_model
        self.use_clean = use_clean

    def set_denoise_model(self, denoise_model):
        self.denoise_model = denoise_model

    def denoise_patches(self, patches):
        batch_size = 100
        for i in tqdm(range(int(len(patches) / batch_size)), file=sys.stdout):
            batch = patches[i * batch_size:(i + 1) * batch_size]
            batch = np.expand_dims(batch, -1)
            batch = np.clip(self.denoise_model.predict(batch).astype(int),
                                        0, 255).astype(np.uint8)[:,:,:,0]
            patches[i*batch_size:(i+1)*batch_size] = batch
        batch = patches[i*batch_size:]
        batch = np.expand_dims(batch, -1)
        batch = np.clip(self.denoise_model.predict(batch).astype(int),
                        0, 255).astype(np.uint8)[:,:,:,0]
        patches[i*batch_size:] = batch
        return patches

    def read_image_file(self, data_dir, train = 1):
        """Return a Tensor containing the patches
        """
        if self.denoise_model and not self.use_clean:
            print('Using denoised patches')
        elif not self.denoise_model and not self.use_clean:
            print('Using noisy patches')
        elif self.use_clean:
            print('Using clean patches')
        sys.stdout.flush()
        patches = []
        labels = []
        counter = 0
        hpatches_sequences = [x[1] for x in os.walk(data_dir)][0]
        if train:
            list_dirs = self.train_fnames
        else:
            list_dirs = self.test_fnames

        for directory in tqdm(hpatches_sequences, file=sys.stdout):
           if (directory in list_dirs):
            for tp in tps:
                if self.use_clean:
                    sequence_path = os.path.join(data_dir, directory, tp)+'.png'
                else:
                    sequence_path = os.path.join(data_dir, directory, tp)+'_noise.png'
                image = cv2.imread(sequence_path, 0)
                h, w = image.shape
                n_patches = int(h / w)
                for i in range(n_patches):
                    patch = image[i * (w): (i + 1) * (w), 0:w]
                    patch = cv2.resize(patch, (32, 32))
                    patch = np.array(patch, dtype=np.uint8)
                    patches.append(patch)
                    labels.append(i + counter)
            counter += n_patches

        patches = np.array(patches, dtype=np.uint8)
        if self.denoise_model and not self.use_clean:
            print('Denoising patches...')
            patches = self.denoise_patches(patches)
        return patches, labels


class DataGeneratorDesc(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, data, labels, num_triplets = 1000000, batch_size=50, dim=(32,32), n_channels=1, shuffle=True):
        # 'Initialization'
        self.transform = False
        self.out_triplets = True
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.data = data
        self.labels = labels
        self.num_triplets = num_triplets
        self.descriptorModel = None
        self.on_epoch_end()

        self.rotationRange = [-30, 30]
        self.zoomRange = [0.8, 1.2]
        self.verticalFlip = True
        self.horizontalFlip = True

    def get_image(self, t):
        def transform_img(img, transform):
            img = np.expand_dims(img, -1)
            if transform:
                IDGgen = IDG()
                params = {}
                params['theta'] = np.random.uniform(self.rotationRange[0], self.rotationRange[1])
                params['zx'] = np.random.uniform(self.zoomRange[0], self.zoomRange[1])
                params['zy'] = np.random.uniform(self.zoomRange[0], self.zoomRange[1])
                
                if self.verticalFlip:
                    params['flip_vertical'] = bool(np.random.randint(2))
                if self.horizontalFlip:
                    params['flip_horizontal'] = bool(np.random.randint(2))

                img = IDGgen.apply_transform(img, params)
                #img = np.squeeze(img)
            return img
        
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a, self.transform).astype(float)
        img_p = transform_img(p, self.transform).astype(float)
        img_n = transform_img(n, self.transform).astype(float)

        # img_a = np.expand_dims(img_a, -1)
        # img_p = np.expand_dims(img_p, -1)
        # img_n = np.expand_dims(img_n, -1)
        #img_a = np.zeros(img_a.shape)
        if self.out_triplets:
            return img_a, img_p, img_n
        else:
            return img_a, img_p

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.triplets) / self.batch_size))
                
    def __getitem__(self, index):
        y = np.zeros((self.batch_size, 1))
        img_a = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_p = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        if self.out_triplets:
            img_n = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        for i in range(self.batch_size):
            t = self.triplets[self.batch_size*index + i]    
            img_a_t, img_p_t, img_n_t = self.get_image(t)
            img_a[i] = img_a_t
            img_p[i] = img_p_t
            if self.out_triplets:
                img_n[i] = img_n_t

        return {'a': img_a, 'p': img_p, 'n': img_n}, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.triplets = generate_triplets(self.labels, self.num_triplets, 32, self.data, self.descriptorModel)

    
    

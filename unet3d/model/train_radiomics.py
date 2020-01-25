import numpy as np
import keras
import cv2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_item_paths,dir_path, batch_size=1, dim=(80,80,80), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_item_paths = list_item_paths
        self.dir_path = dir_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_item_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_item_paths_temp = [self.list_item_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_item_paths_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_item_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_item_paths_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size, self.n_channels, *self.dim))

        # Generate data
        for i, item_path in enumerate(list_item_paths_temp):
            # Store sample

            archive = np.load(self.dir_path + item_path)
            scan = archive['scan']
            mask = archive['mask']



            scan = cv2.resize(scan, dsize=(80, 80))[:,:,6:-6]
            mask = cv2.resize(mask.astype(np.uint8), dsize=(80, 80))[:,:,6:-6]

            X[i,] = np.expand_dims(scan, axis=0)

            # Store class
            y[i] = np.expand_dims(mask, axis=0)

        return X, y


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    archives = os.listdir('/home/david/Documents/chall_owkin/train/images/')

    gen = DataGenerator(archives, "/home/david/Documents/chall_owkin/train/images/")


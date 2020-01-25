from unittest import TestCase

from unet3d.model import unet_model_3d
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model

class TestModel(TestCase):
    def test_batch_normalization(self):
        model = unet_model_3d(input_shape=(1, 16, 16, 16), depth=2, deconvolution=True, metrics=[], n_labels=1,
                              batch_normalization=True)

        layer_names = [layer.name for layer in model.layers]

        for name in layer_names[:-3]:  # exclude the last convolution layer
            if 'conv3d' in name and 'transpose' not in name:
                self.assertIn(name.replace('conv3d', 'batch_normalization'), layer_names)

model = isensee2017_model((1, 80, 80, 80), n_labels=1)

model_old = load_old_model('../isensee_2017_model.h5')

import numpy as np
import cv2

archive = np.load('/home/david/Documents/chall_owkin/train/images/patient_002.npz')
scan = archive['scan']
mask = archive['mask']

scan = cv2.resize(scan, dsize=(80, 80))[:,:,6:]

print(model.summary())

print('\n',scan.shape)

# pred = model.predict(np.expand_dims(np.expand_dims(scan, axis=0), axis=1))
#data processing

import tensorflow as tf
from glob import glob
import numpy as np
from IPython.display import display, Image
import tensorflow_io as tfio
from tensorflow.io import FixedLenFeature, VarLenFeature
import matplotlib.pyplot as plt
import os

def decode_fn(record_bytes):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(record_bytes, features) 
    image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
    return image

def preprocess_image(image):
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_and_preprocess_dataset(file_paths):
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(decode_fn)
    dataset = dataset.map(preprocess_image)
    return dataset.batch(4)

if __name__== "__main__":
    cleaned_dir = r'GAN_tf_records/records/cleaned'
    if not os.path.isdir(cleaned_dir):
        os.makedirs(cleaned_dir)
    
    monet_dirpath = r'CycleGanProject\GAN_tf_records\records\monet_tfrec'
    photo_dirpath = r'CycleGanProject\GAN_tf_records\records\photo_tfrec'
    monet_file = tf.io.gfile.glob(monet_dirpath + '/*.tfrec')
    photo_file = tf.io.gfile.glob(photo_dirpath + '/*.tfrec')
    print('Monet # of files:' , len(monet_file))
    print('Photo # of files:' , len(photo_file))
 
    monet_normalized = load_and_preprocess_dataset(monet_file)
    photo_normalized = load_and_preprocess_dataset(photo_file)

#The code used to get the shape of the tensor images
#Output: shape=(4, 256, 256, 3), dtype=float32 <-- 4 batches, 256 x 256 matrix, 3 channels (rgb)
    #for element in monet_normalized:
    #    print(element)

input_channels = 3
kernel_size = 3
output_channels = 64

generator = Generator(input_channels, output_channels, kernel_size)




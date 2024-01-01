#Upload files

import tensorflow as tf
from glob import glob
from tensorflow import keras
import numpy as np
from IPython.display import display, Image
from tensorflow.io import FixedLenFeature, VarLenFeature
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

if __name__== "__main__":
    cleaned_dir = r'GAN_tf_records/records/cleaned'
    if not os.path.isdir(cleaned_dir):
        os.makedirs(cleaned_dir)
    
    
    monet_dirpath = 'GAN_tf_records/records/monet_tfrec'
    photo_dirpath = 'GAN_tf_records/records/photo_tfrec'
    monet_file = tf.io.gfile.glob(monet_dirpath + '/*.tfrec')
    photo_file = tf.io.gfile.glob(photo_dirpath + '/*.tfrec')
    print('Monet # of files:' , len(monet_file))
    print('Photo # of files:' , len(photo_file))
 
    monet_dataset = tf.data.TFRecordDataset(monet_file) #A Dataset comprising records from one or more TFRecord files.
    photo_dataset = tf.data.TFRecordDataset(photo_file)

    #This is where the features will go so that we can parse the images.
    def decode_fn(record_bytes):
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'target': tf.io.FixedLenFeature([], tf.string),
        }
        parsed_features = tf.io.parse_single_example(record_bytes, features)
        image = tf.image.decode_jpeg(parsed_features['image'])
        image = tf.cast(image, tf.float32)
        normalized_image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
        return normalized_image
    
    batch_size = 32
    
    #Creates a map that attaches the features to values. Like a dictionary.
    monet_dataset_map = monet_dataset.map(decode_fn).batch(batch_size)
    photo_dataset_map = photo_dataset.map(decode_fn).batch(batch_size)

    type(monet_dataset_map)
    
    def show_images(dataset, num_images=10):
        plt.figure(figsize=(15, 15))  # Adjusted figure size for better visibility
        for images in dataset.take(1):  # Take one batch
            num_images = min(num_images, images.shape[0])  # Ensure we don't exceed the batch size
            for i in range(num_images):  # Iterate over the first 'num_images' images in the batch
                ax = plt.subplot(1, num_images, i + 1)
                plt.imshow(images[i])
                plt.axis("off")
        plt.show()
    
    show_images(monet_dataset_map)
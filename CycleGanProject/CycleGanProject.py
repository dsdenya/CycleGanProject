#Upload files

import tensorflow as tf
from glob import glob
from tensorflow import keras
import numpy as np
from IPython.display import display, Image
import tensorflow_io as tfio
from tensorflow.io import FixedLenFeature, VarLenFeature
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

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
 
    monet_dataset = tf.data.TFRecordDataset(monet_file) #Creates a tensorflow Dataset comprising records from one or more TFRecord files.
    photo_dataset = tf.data.TFRecordDataset(photo_file)


    #An Example is a standard proto storing data for training and inference. Also, this will get the 'features' of the tfrecord so that you can parse it
    #Iterates through the first 10 records of each dataset, parsing them into tf.train.Example objects to inspect their contents.
    for raw_record in monet_dataset.take(10):
      monet = tf.train.Example()
      monet.ParseFromString(raw_record.numpy())
      
    for photo_raw_record in photo_dataset.take(10):
      photo = tf.train.Example()
      photo.ParseFromString(photo_raw_record.numpy())

    #This is where the features will go so that we can parse the images.
    #The decode_fn function extracts the image field from each record. This field contains JPEG-encoded image data.
    #The JPEG data is then decoded into an image tensor using tf.image.decode_jpeg within the tensor_normalization function
    
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
        # Convert image to float32 and normalize
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def load_and_preprocess_dataset(file_paths):
        dataset = tf.data.TFRecordDataset(file_paths)
        dataset = dataset.map(decode_fn)
        dataset = dataset.map(preprocess_image)
        return dataset

    monet_normalized = load_and_preprocess_dataset(monet_file).batch(3)
    photo_normalized = load_and_preprocess_dataset(photo_file).batch(3)
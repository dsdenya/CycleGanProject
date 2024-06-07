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

# The origins of the features (how the dataset was parsed in the first place) is shown in the past commit which was this:
#for raw_record in monet_dataset.take(10):
#      monet = tf.train.Example()
#      monet.ParseFromString(raw_record.numpy())
      
#for photo_raw_record in photo_dataset.take(10):
#      photo = tf.train.Example()
#     photo.ParseFromString(photo_raw_record.numpy())
# That's how we got the features below but since i know what it is now..just took it out

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
    # Convert image to float32 and normalize because it was originally Uint8 
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_and_preprocess_dataset(file_paths):
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(decode_fn)
    dataset = dataset.map(preprocess_image)
    return dataset.batch(3)

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

# ****My personal for the future refresher summary notes****: 
# So, in the load_and_preprocess_dataset, we'recreating a tfrecord dataset from the file paths. 
# Then in dataset = dataset.map(decode_fn), we are taking the dataset contents passing that to decode_fn to get specifically the 
# image feature (that is now an an image tensor btw). And then once we have the image we normalize it in dataset.dataset.map(preprocess_image) and
# then we return batches or groups of the images.
# The previous code was "This is the 'before' code. In this code, I had memory allocation issues that were exceeding the memory by 10%. 
# This was occuring within the  decode_fn and tensor_normalization functions. The next priority to optimize this code is this so it doesn't exceed memory problems again...
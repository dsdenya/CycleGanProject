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
        return tf.io.parse_single_example(record_bytes, features)
    
    #Creates a map that attaches the features to values. Like a dictionary.
    #Uses the map function to apply decode_fn to each element of the Monet and photo datasets, 
    #effectively parsing the records into dictionaries with the expected features.

    monet_dataset_map = monet_dataset.map(decode_fn)
    photo_dataset_map = photo_dataset.map(decode_fn)

    type(monet_dataset_map)
    
    
    #Make a normalization and tensor function and put it in a list and return it.
    #tensor_normalization function takes a dataset of parsed records, decodes the JPEG image, casts it to tf.float32, and normalizes it.
    #The function returns a new dataset of normalized image tensors.
    #Within the tensor_normalization function, the features['image'] contains the JPEG image data as a serialized string.
    #The tf.image.decode_jpeg(features['image'].numpy()) function call decodes the serialized string back into an image tensor..
    def tensor_normalization(dataset_map):
      images_normalized = []
      for features in dataset_map:

        tensor = tf.image.decode_jpeg(features['image'].numpy())
        tensor = tf.cast(tensor, tf.float32) # Since the original type of image_tensor was uint8, we need to change it to tf.float32 so it can be compatible to perform arithemitc with 127.5
        normalized = (tensor - np.amin(tensor))/ (np.amax(tensor) - np.amin(tensor)) #Normalization formula
        images_normalized.append(normalized)
      
      images_normalized_dt = tf.data.Dataset.from_tensor_slices(images_normalized)
      return images_normalized_dt.batch(3)
      #return tf.data.Dataset.from_tensor_slices(images_normalized)
    #Applies the normalization function to both Monet and photo datasets, resulting in monet_normalized
    #and photo_normalized datasets containing normalized image tensors.So, basically, we have two 
    #TensorFlow datasets (monet_normalized and photo_normalized) containing normalized image tensors derived from the original TFRecord files.  
    monet_normalized = tensor_normalization(monet_dataset_map)
    photo_normalized = tensor_normalization(photo_dataset_map)
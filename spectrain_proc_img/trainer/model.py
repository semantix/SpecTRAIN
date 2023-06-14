import datetime
import os
import shutil
import numpy as np
import tensorflow as tf
import hypertune
import numpy as np
from google.cloud import bigquery, storage
from google.oauth2 import credentials
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Softmax)

def get_blob(blobs):
    for blob in blobs:
        yield blob
        
def get_image_paths(image_input_dir):
    # initialize the GCS client
    image_bucket = image_input_dir.split('/')[2]
    prefix_dir = '/'.join(image_input_dir.split('/')[3:])
    prefix_dir = prefix_dir[1:]
    storage_client = storage.Client()
    # get the storage bucket
    bucket = storage_client.get_bucket(image_bucket)

    image_paths=[]
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(image_bucket, prefix=prefix_dir)
    
    for blob in get_blob(blobs):
        if "output" in blob.name:
            image_paths.append('gs://spectrain/'+blob.name)
    return image_paths

def load_images(imagePath):
    # read the image from disk, decode it, convert the data type to
    # floating point, and resize it
    
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (256,256))
    # parse the class label from the file path
    label = tf.strings.split(imagePath, os.path.sep)[-2]
    if label=='positive':
        label=1
    else:
        label=0
    # return the image and the label
    return (image, label)

    # return the image and the label
    return (image, label)

def load_dataset(images_dir, batch_size, training):
    
    filePaths = get_image_paths(image_input_dir=images_dir)
    ds = tf.data.Dataset.from_tensor_slices(filePaths)
    ds = (ds
        .map(load_images)
        .cache()
        .shuffle(len(filePaths))
        .batch(batch_size)
    )

    if training:
        return ds.repeat()
    else:
        return ds

def build_model(filter_size_1, filter_size_2, kernel_size, pool_kernel_size, hidden_units_1, hidden_units_2):
    model = Sequential()
    model.add(Conv1D(filter_size_1, kernel_size=kernel_size, activation='relu', input_shape=(256, 256)))
    model.add(Conv1D(filter_size_1, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_kernel_size))
    model.add(Conv1D(filter_size_2, kernel_size=kernel_size,activation='relu'))
    model.add(MaxPooling1D(pool_kernel_size))
    model.add(Flatten())
    model.add(Dense(hidden_units_1, activation='relu'))
    model.add(Dense(hidden_units_1, activation='relu'))
    model.add(Dense(hidden_units_2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    
    return model

    
# Instantiate the HyperTune reporting object
hpt = hypertune.HyperTune()

# Reporting callback
class HPTCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        global hpt
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='auc',
            metric_value=logs['val_auc'],
            global_step=epoch)
        
        
def train_and_evaluate(args):
    model = build_model(filter_size_1=args["filt_size1"], filter_size_2=args["filt_size2"], 
                        kernel_size=args['ksize'], pool_kernel_size=args['pool_ksize']
                        , hidden_units_1=args['nnsize_1'], hidden_units_2=args['nnsize_2'])

    trainds = load_dataset(args["train_data_path"], args["batch_size"], training=True)

    evalds = load_dataset(args["eval_data_path"], args["batch_size"], training=False)
    
    if args["eval_steps"]:
        evalds = evalds.take(count=args["eval_steps"])

    num_batches = args["batch_size"] * args["num_epochs"]
    steps_per_epoch = args["train_examples"] // args["batch_size"]
    checkpoint_path = os.path.join(args["output_dir"], "checkpoints/spectrain_proc_img")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_weights_only=True)

    history = model.fit(
        trainds,
        validation_data=evalds,
        epochs=args["batch_size"],
        steps_per_epoch=steps_per_epoch,
        verbose=2,
        callbacks=[cp_callback, HPTCallback()])
    
    EXPORT_PATH = os.path.join(
        args["output_dir"], datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    tf.saved_model.save(
        obj=model, export_dir=EXPORT_PATH)  # with default serving function
    
    print("Exported trained model to {}".format(EXPORT_PATH))

from __future__ import print_function
import os
import numpy as np
import pandas as pd
import imp
import datetime
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    pass    
import glob
import sys
import tqdm
import argparse
import pathlib
import tensorflow as tf
from tensorflow.keras import layers, models
from interaction import LEIA
from data import H5Data

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
if os.path.isdir('/data/shared/hls-fml/'):
    test_path = '/data/shared/hls-fml/NEWDATA/'
    train_path = '/data/shared/hls-fml/NEWDATA/'
elif os.path.isdir('/eos/project/d/dshep/hls-fml/'):
    test_path = '/eos/project/d/dshep/hls-fml/'
    train_path = '/eos/project/d/dshep/hls-fml/'

N = 100 # number of particles
n_targets = 5 # number of classes
n_features = 4 # number of features per particles
save_path = 'models/2/'
best_path = save_path + '/best/'
batch_size = 256
n_epochs = 10

files = glob.glob(train_path + "/jetImage*_{}p*.h5".format(N))
num_files = len(files)
files_val = files[:int(num_files*0.2)] # take first 20% for validation
files_train = files[int(num_files*0.2):] # take rest for training
    
data_train = H5Data(batch_size = batch_size,
                    cache = None,
                    preloading=0,
                    features_name='jetConstituentList', 
                    labels_name='jets',
                    spectators_name=None)
data_val = H5Data(batch_size = batch_size,
                  cache = None,
                  preloading=0,
                  features_name='jetConstituentList', 
                  labels_name='jets',
                  spectators_name=None)

            
# Define loss function
def loss(model, x, y):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    y_ = model(x)
    return cce(y_true=y, y_pred=y_)

def main(args):
    """ Main entry point of the app """
    data_train.set_file_names(files_train)
    data_val.set_file_names(files_val)
    
    n_val=data_val.count_data()
    n_train=data_train.count_data()

    print("val data:", n_val)
    print("train data:", n_train)

    net_args = (N, n_targets, n_features, args.hidden)
    net_kwargs = {"fr_activation": 0, "fo_activation": 0, "fc_activation": 0}
    
    gnn = LEIA(*net_args, **net_kwargs)
    gnn.build(input_shape=(None, N, n_features))
    
    # gnn.summary() # Doens't work. Seems like a common TF 2.0 issue: 
    # https://github.com/tensorflow/tensorflow/issues/22963 
    # https://stackoverflow.com/questions/58182032/you-tried-to-call-count-params-on-but-the-layer-isnt-built-tensorflow-2-0

    #### Start training ####
    
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    val_loss_results = []
    val_accuracy_results = []
    
    # Log directory for Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    pathlib.Path(train_log_dir).mkdir(parents=True, exist_ok=True)  
    pathlib.Path(test_log_dir).mkdir(parents=True, exist_ok=True)  

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    
    best_loss = 100
    for epoch in range(n_epochs):
        
        # Tool to keep track of the metrics
        epoch_loss_avg = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
        val_epoch_loss_avg = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        val_epoch_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

        # Training
        for sub_X, sub_Y in tqdm.tqdm(data_train.generate_data(),total = n_train/batch_size):
#            print(f"sub_X: {sub_X.shape}")
#            print(f"sub_Y: {sub_Y.shape}")
            training = sub_X.astype(np.float32)[:,:,[3,0,1,2]]
            target = sub_Y.astype(np.float32)[:,-6:-1]
            def grad(model, input_par, targets):
                with tf.GradientTape() as tape:
                    loss_value = loss(model, input_par, targets)
                return loss_value, tape.gradient(loss_value, model.trainable_variables)
            
            # Define optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            # Compute loss and gradients
            loss_value, grads = grad(gnn, training, target)

            # Update the gradients
            optimizer.apply_gradients(zip(grads, gnn.trainable_variables))

            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            epoch_accuracy(target, tf.nn.softmax(gnn(training)))

        # Validation
        for sub_X, sub_Y in tqdm.tqdm(data_val.generate_data(),total = n_val/batch_size):
            training = sub_X.astype(np.float32)[:,:,[3,0,1,2]]
            target = sub_Y.astype(np.float32)[:,-6:-1]
            
            # Compute the loss
            loss_value = loss(gnn, training, target)
            
            # Track progress
            val_epoch_loss_avg(loss_value)
            val_epoch_accuracy(target, tf.nn.softmax(gnn(training)))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        val_loss_results.append(val_epoch_loss_avg.result())
        val_accuracy_results.append(val_epoch_accuracy.result())
       
        # Save best epoch only
        if best_loss > val_epoch_loss_avg.result():
            best_loss = val_epoch_loss_avg.result()

            # Save the model after training
            pathlib.Path(best_path).mkdir(parents=True, exist_ok=True)  
            gnn.save_weights(best_path, save_format='tf')

        # Logs for tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_loss_avg.result(), step=epoch)
            tf.summary.scalar('accuracy', epoch_accuracy.result(), step=epoch)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', val_epoch_loss_avg.result(), step=epoch)
            tf.summary.scalar('accuracy', val_epoch_accuracy.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}%, Test Loss: {}, Test Accuracy: {}%'
        print (template.format(epoch+1,
                         epoch_loss_avg.result(), 
                         epoch_accuracy.result()*100,
                         val_epoch_loss_avg.result(), 
                         val_epoch_accuracy.result()*100))

        # Reset metrics every epoch
        epoch_loss_avg.reset_states()
        val_epoch_loss_avg.reset_states()
        epoch_accuracy.reset_states()
        val_epoch_accuracy.reset_states()

    # Save the model after training
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)  
    gnn.save_weights(save_path, save_format='tf')

def evaluate(args):
    net_args = (N, n_targets, n_features, args.hidden)
    net_kwargs = {"fr_activation": 0, "fo_activation": 0, "fc_activation": 0}
    
    gnn = LEIA(*net_args, **net_kwargs)
    gnn.build(input_shape=(None, N, n_features))
    gnn.load_weights(best_path)
    
    data_val.set_file_names(files_val)
    n_val=data_val.count_data()
        
    epoch_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy('accuracy')

    # Validation
    for sub_X, sub_Y in tqdm.tqdm(data_val.generate_data(),total = n_val/batch_size):
        training = sub_X.astype(np.float32)[:,:,[3,0,1,2]]
        target = sub_Y.astype(np.float32)[:,-6:-1]
        
        # Compute the loss
        loss_value = loss(gnn, training, target)
        
        # Track progress
        epoch_loss(loss_value)
        epoch_accuracy(target, tf.nn.softmax(gnn(training)))
        
    template = 'Loss: {}, Accuracy: {}%'
    print (template.format(epoch_loss.result(), 
                     epoch_accuracy.result()*100))

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    
    # Required positional arguments
    
    # Optional arguments
    parser.add_argument("--hidden", type=int, action='store', dest='hidden', default = 16, help="hidden parameter")
    parser.add_argument("--eval", action='store_true', dest='evaluate', default = False, help="only run evaluation")

    args = parser.parse_args()
    if not args.evaluate: main(args)
    else: evaluate(args)

import glob
import os

import tensorflow as tf
import numpy as np
from tensorboard import program
from Dataloader import process_path, load_dataset
from models import Autoencoder, train, loss

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)



learning_rate = 0.0005
batch_size = 256
epochs = 100

autoencoder = Autoencoder()
opt = tf.optimizers.Adam(learning_rate=learning_rate)



path = '/home/jasper/Documents/Shading_Dataset/Only Iris/images/'
files = tf.data.Dataset.list_files('/media/jasper/7834B4AE34B470AC/data_egohands/_LABELLED_SAMPLES/*/*.jpg', shuffle=True)
train_set, test_set = load_dataset('/media/jasper/7834B4AE34B470AC/data_egohands/_LABELLED_SAMPLES/*/*.jpg', 64)

def train2():
    print(logical_gpus)
    print("numvber of steps : " + str(train.__sizeof__()))
    #empty logdir if old log in there
    if not (len(os.listdir('tmp')) == 0):
        files = glob.glob(path+"/*")
        for f in files:
            os.remove(f)
    #create tensorboard
    writer = tf.summary.create_file_writer('tmp')
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'tmp'])
    url = tb.launch()
    print(url)
    with writer.as_default():
        try:
            # Specify an invalid GPU device
            with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):

                with tf.summary.record_if(True):
                    ctr = 0
                    test_ctr = 0
                    for epoch in range(epochs):
                            print("We are at " + str(epoch) + " out of " +str(epochs))
                            for step, (batch_features, label) in enumerate(test_set):
                                train(autoencoder, opt, batch_features,label)
                                loss_values = loss(autoencoder, batch_features,label)
                                print(str(step) + "  :  " + str(loss_values))
                                rec = autoencoder(tf.constant(batch_features))
                                tf.summary.scalar('test_loss', loss_values,step=test_ctr)
                                if step % 50 == 0 :
                                    tf.summary.image('original', batch_features, max_outputs=1, step=test_ctr)
                                    tf.summary.image('reconstructed', rec, max_outputs=1,step = test_ctr)
                                    tf.summary.image('goal', label, max_outputs=1, step=test_ctr)
                                test_ctr += 1

                            for step, (batch_features, label) in enumerate(train_set):
                                train(autoencoder, opt, batch_features,label)
                                loss_values = loss(autoencoder, batch_features,label)
                                print(str(step) + "  :  " + str(loss_values))
                                tf.summary.scalar('train_loss', loss_values,step=ctr)
                                ctr+=1

                            autoencoder.encoder.save_weights('encoder.h5')
                            autoencoder.decoder.save_weights('decoder.h5')
        except RuntimeError as e:
            print(e)


train2()
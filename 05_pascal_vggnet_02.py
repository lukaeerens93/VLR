from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial

from eval import compute_map


tf.logging.set_verbosity(tf.logging.INFO)
CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]



vgg_mapping = {
    'vgg_16/conv1/conv1_1/biases':'conv2d/bias',
    'vgg_16/conv1/conv1_1/weights':'conv2d/kernel',
    'vgg_16/conv1/conv1_2/biases':'conv2d_1/bias',
    'vgg_16/conv1/conv1_2/weights':'conv2d_1/kernel',
 
    'vgg_16/conv2/conv2_1/biases':'conv2d_2/bias',
    'vgg_16/conv2/conv2_1/weights':'conv2d_2/kernel',
    'vgg_16/conv2/conv2_2/biases':'conv2d_3/bias',
    'vgg_16/conv2/conv2_2/weights':'conv2d_3/kernel',

    'vgg_16/conv3/conv3_1/biases':'conv2d_4/bias',
    'vgg_16/conv3/conv3_1/weights':'conv2d_4/kernel',
    'vgg_16/conv3/conv3_2/biases':'conv2d_5/bias',
    'vgg_16/conv3/conv3_2/weights':'conv2d_5/kernel',
    'vgg_16/conv3/conv3_3/biases':'conv2d_6/bias',
    'vgg_16/conv3/conv3_3/weights':'conv2d_6/kernel',

    'vgg_16/conv4/conv4_1/biases':'conv2d_7/bias',
    'vgg_16/conv4/conv4_1/weights':'conv2d_7/kernel',
    'vgg_16/conv4/conv4_2/biases':'conv2d_8/bias',
    'vgg_16/conv4/conv4_2/weights':'conv2d_8/kernel',
    'vgg_16/conv4/conv4_3/biases':'conv2d_9/bias',
    'vgg_16/conv4/conv4_3/weights':'conv2d_9/kernel',

    'vgg_16/conv5/conv5_1/biases':'conv2d_10/bias',
    'vgg_16/conv5/conv5_1/weights':'conv2d_10/kernel',
    'vgg_16/conv5/conv5_2/biases':'conv2d_11/bias',
    'vgg_16/conv5/conv5_2/weights':'conv2d_11/kernel',
    'vgg_16/conv5/conv5_3/biases':'conv2d_12/bias',
    'vgg_16/conv5/conv5_3/weights':'conv2d_12/kernel',

    'vgg_16/fc6/biases':'conv2d_13/bias',
    'vgg_16/fc6/weights':'conv2d_13/kernel',
    'vgg_16/fc7/biases':'conv2d_14/bias',
    'vgg_16/fc7/weights':'conv2d_14/kernel'
    }


               
def cnn_model_fn(features, labels, mode, num_classes=20):

    # Input Layer
    crop_height = 224
    crop_width = 224
    if mode == tf.estimator.ModeKeys.TRAIN:
        def augment(img):
            return tf.random_crop(value=tf.image.random_flip_left_right(img),size = [crop_height, crop_width, 3])
        input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
        input_layer=tf.map_fn(augment,input_layer)   
    else:
        input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])
 
    # Image Loggin to Tensorboard
    tf.summary.image("Images",input_layer)

    conv1_1 = tf.layers.conv2d(inputs=input_layer,filters=64,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    conv1_2 = tf.layers.conv2d(inputs=conv1_1,filters=64,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=2, strides=2,name = 'pool1')

    conv2_1 = tf.layers.conv2d(inputs=pool1,filters=128,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    conv2_2 = tf.layers.conv2d(inputs=conv2_1,filters=128,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=2, strides=2, name = 'pool2')

    conv3_1 = tf.layers.conv2d(inputs=pool2,filters=256,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    conv3_2 = tf.layers.conv2d(inputs=conv3_1,filters=256,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    conv3_3 = tf.layers.conv2d(inputs=conv3_2,filters=256,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=2, strides=2,name = 'pool3')

    conv4_1 = tf.layers.conv2d(inputs=pool3,filters=512,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    conv4_2 = tf.layers.conv2d(inputs=conv4_1,filters=512,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    conv4_3 = tf.layers.conv2d(inputs=conv4_2,filters=512,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=2, strides=2,name = 'pool4')

    conv5_1 = tf.layers.conv2d(inputs=pool4,filters=512,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    conv5_2 = tf.layers.conv2d(inputs=conv5_1,filters=512,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    conv5_3 = tf.layers.conv2d(inputs=conv5_2,filters=512,strides = 1,kernel_size=3,padding="same",activation=tf.nn.relu)
    pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=2, strides=2,name = 'pool5')
    p5s = pool5.get_shape()

    fc6 = tf.layers.conv2d(inputs=pool5, filters=4096, strides = 1 ,activation=tf.nn.relu,kernel_size = 7,padding ="valid")
    dropout1 = tf.layers.dropout(inputs=fc6, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    fc7 = tf.layers.conv2d(inputs=dropout1, filters=4096,strides = 1,activation=tf.nn.relu, kernel_size = 1, padding = "valid")
    dropout2 = tf.layers.dropout(inputs=fc7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dropout2 = tf.reshape(dropout2,[-1,4096])

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=20,name = 'fc8')

    # Generate predictions (for PREDICT and EVAL mode) and add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
    predictions = {"classes": tf.argmax(input=logits, axis=1),"probabilities": tf.sigmoid(logits, name="sigmoid_tensor"), "pool5": pool5, "fc7": fc7}


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy (multi_class_labels=labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        decayed_lr = tf.train.exponential_decay(0.001, global_step = tf.train.get_global_step() , decay_steps = 1000, decay_rate= 0.5, staircase=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=decayed_lr,momentum = 0.9)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




def load_pascal(data_dir, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that 
            are ambiguous.
    """
    filename = data_dir+"/ImageSets/Main/" + split+".txt"
    files = open(filename).read()
    data_splitup = files.split('\n')[:-1]
    
    N = len(data_splitup)

    labels = np.zeros([N, len(CLASS_NAMES)],dtype = np.int32)
    weights = np.zeros([N, len(CLASS_NAMES)], dtype =np.int32)
    for class_number in range(len(CLASS_NAMES)):
        class_files = []
        filename2 = data_dir+"/ImageSets/Main/" + CLASS_NAMES[class_number] + "_" + split+".txt"
        files2 = open(filename2).read()
        data_splitup2 = files2.split('\n')[:-1]

        for file in data_splitup2:
            file = file.split(" ")
            class_files.append([file[0],file[-1]])

        for image_number in range(N):
            if(int(class_files[image_number][1]) != 0):
                weights[image_number, class_number] = 1
            if(int(class_files[image_number][1]) == 1):
                labels[image_number, class_number] = 1

    if split == 'trainval':
        images = np.empty([N,256,256,3], dtype = np.float32)
        for i in range(N):
            images[i,:,:,:] = Image.open(data_dir + '/JPEGImages/' + data_splitup[i]+ '.jpg').resize((256,256), Image.BILINEAR)
    else:
        images = np.empty([N, 224, 224, 3], dtype = np.float32)
        for i in range(N):
            images[i,:,:,:] = Image.open(data_dir +'/JPEGImages/'+data_splitup[i]+'.jpg').resize((256,256), Image.BILINEAR).crop((16,16,240,240))

    return images, labels, weights
    


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args




def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr




class vgg_16(tf.train.SessionRunHook):
    def begin(self):
        tf.contrib.framework.init_from_checkpoint('vgg_16.ckpt',vgg_mapping)




def main():
    args = parse_args()
    
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(model_fn=partial(cnn_model_fn,num_classes=train_labels.shape[1]),model_dir="./q4_vgg")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data, "w": train_weights},y=train_labels,batch_size=10,num_epochs=None,shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data, "w": eval_weights},y=eval_labels,num_epochs=1,shuffle=False)

    pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
    np.save('vgg.npy', pred)
    pred = np.stack([p['probabilities'] for p in pred])
    rand_AP = compute_map(eval_labels, np.random.random(eval_labels.shape),eval_weights, average=None)
    print('Random AP: {} mAP'.format(np.mean(rand_AP)))
    gt_AP = compute_map(eval_labels, eval_labels, eval_weights, average=None)
    print('GT AP: {} mAP'.format(np.mean(gt_AP)))
    AP = compute_map(eval_labels, pred, eval_weights, average=None)
    meanAP = np.mean(AP)
    print('Obtained {} mAP'.format(np.mean(AP)))
    print('per class:')
    for cid, cname in enumerate(CLASS_NAMES):
        print('{}: {}'.format(cname, _get_el(AP, cid)))
    

if __name__ == "__main__":
    main()

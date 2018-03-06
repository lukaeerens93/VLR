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
import os

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


def cnn_model_fn(features, labels, mode, num_classes=20):
    if mode == tf.estimator.ModeKeys.TRAIN:
        def augment(img):
            return tf.random_crop(value=tf.image.random_flip_left_right(img),size = [224, 224, 3])
        input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
        input_layer=tf.map_fn(augment,input_layer)   
    else:
        input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    p2s = pool2.get_shape()
    
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, int(p2s[1]) * int(p2s[2]) * int(p2s[3])])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=20)

    # Generate predictions (for PREDICT and EVAL mode) and add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
    predictions = {"classes": tf.argmax(input=logits, axis=1),"probabilities": tf.sigmoid(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy (
        multi_class_labels=labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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
    # Wrote this function
 
    # Find all of the relevant files in the pascal directory
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



def main():

    args = parse_args()
    # Load training and eval data
    eval_data, eval_labels, eval_weights = load_pascal(args.data_dir, split='test')
    train_data, train_labels, train_weights = load_pascal(args.data_dir, split='trainval')

    test_data = (eval_data,eval_labels,eval_weights)
    trainval = (train_data,train_labels, train_weights)
    pascal_classifier = tf.estimator.Estimator(model_fn=partial(cnn_model_fn,num_classes=train_labels.shape[1]),model_dir="./pascal_MNIST")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data, "w": train_weights}, y=train_labels, batch_size=10, num_epochs=None, shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data, "w": eval_weights}, y=eval_labels, num_epochs=1, shuffle=False)
    
    session = tf.Session() 
    file_writer = tf.summary.FileWriter('./pascal_MNIST/Train',session.graph)  

    for i in range(10):

        pascal_classifier.train(input_fn=train_input_fn, steps=100, hooks=[logging_hook])
        # Evaluate the model and print results
        pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
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
        
        summary_map = tf.Summary(value = [tf.Summary.Value(tag='mAP',simple_value=meanAP)])
        file_writer.add_summary(summary_map,100*i)



if __name__ == "__main__":
    main()

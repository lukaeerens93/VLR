from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sklearn
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

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






#==== 
# Reference : https://github.com/oreillymedia/t-SNE-tutorial
def scatter(x, colors):
    palette = np.array(sns.hls_palette(20))
    ax.scatter(x[:,0], x[:,1], lw=0, s=40, alpha=0.4, c=palette[colors.astype(np.int)])
    plt.show()


eval_data, eval_labels,eval_weights = load_pascal('VOCdevkit/VOC2007', split='test')

eval_labels = np.random.permutation(eval_labels)[:1000]
labels = np.zeros(1000)
i = 0
while (i < 1000):
    labels[i] = np.average(np.where(eval_labels[i]==1))
    i = i + 1

alex_net = np.load('AlexNet.npy',encoding='bytes')
alex_net = np.array(alex_net)
features = np.vstack([x['fc9'].flatten() for x in alex_net])
tsne_proj =TSNE(n_components=2).fit_transform(features)
sns.palplot(sns.hls_palette(20))
scatter(tsne_proj,labels)


import numpy as np
import random
from numpy.linalg import norm
import cv2

alex = np.load('Alexnet.npy')
vgg = np.load('vgg.npy')

p3_alex = np.stack([p['pool3'] for p in alex])
size = p3_alex.shape
p3_alex = np.reshape(p3_alex,[-1,size[1]*size[2]*size[3]])
fc7_alex = np.stack([p['fc7'] for p in alex])
size = fc7_alex.shape
fc7_alex = np.reshape(fc7_alex,[-1,size[-1]])

p5_vgg = np.stack([p['pool5'] for p in vgg])
size = p5_vgg.shape
p5_vgg = np.reshape(p5_vgg,[-1,size[1]*size[2]*size[3]])
fc7_vgg = np.stack([p['fc7'] for p in vgg])
size = fc7_vgg.shape
fc7_vgg = np.reshape(fc7_vgg,[-1,size[-1]])

def save_neighbors(neighbor_dict,layer_name):
	for i,(image,neighbor) in enumerate(neighbor_dict.items()):
		image_array = cv2.imread("VOCdevkit/VOC2007/JPEGImages/" + str(image).zfill(6) + ".jpg")
		cv2.imwrite(str(layer_name)+str(i).zfill(2)+"1.jpg",image_array)
		neighbor_array = cv2.imread("VOCdevkit/VOC2007/JPEGImages/" + str(neighbor).zfill(6) + ".jpg")
		cv2.imwrite(str(layer_name)+str(i).zfill(2)+"2.jpg",neighbor_array)

def getneighbors(layer,layer_name,indices):
	test_images = layer[indices,:]
	neighbor_dict = {}
	for (i,test_image) in enumerate(test_images):
		min_dist = float('inf')
		for (j,potential_neighbor) in enumerate(layer):
			dist = norm(test_image - potential_neighbor)
			if(dist!=0 and dist<min_dist):
				neighbor_dict[indices[i]] = j
				min_dist = dist		
	save_neighbors(neighbor_dict, layer_name)
	print(neighbor_dict)

indices = np.random.choice(4952,10,replace=True)
layers = [p3_alex,fc7_alex, p5_vgg,fc7_vgg]
layer_names = ["Pool5_AlexNet","FC7_AlexNet","Pool5_VGGNet","FC7_VGGNet"]
for i,layer in enumerate(layers):
	print ("layer name: " + str(layer_names[i]))
	getneighbors(layer,layer_names[i],indices)
	print ("")

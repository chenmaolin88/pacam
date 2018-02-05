
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import math
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# In[13]:


class MotionDetector:

    tmp = []  #计算熵值的临时变量
    fgbg = None
    detection_graph = None
    detection_scores = None
    detection_classes = None
    image_tensor = None
    sess = None
    human_score_threshold = 0.1 #只关注概率大于1０％的人类
    entropy_low_threshold =  0.03 #图像差异熵的最低阀值
    entropy_high_threshold =  1 #图像差异熵的最高阀值

    def __init__(self):
        sys.path.append("..")
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        from utils import label_map_util
        from utils import visualization_utils as vis_util
        MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
        PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
        NUM_CLASSES = 90
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

	config = tf.ConfigProto(device_count = {'GPU': 0}) 
	self.sess=tf.Session(graph=self.detection_graph,config=config) 
	# Definite input and output Tensors for detection_graph
	self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
	# Each box represents a part of the image where a particular object was detected.
	#detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
	# Each score represent how level of confidence for each of the objects.
	# Score is shown on the result image, together with the class label.
	self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
	self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')


        self.fgbg = cv2.BackgroundSubtractorMOG()
        for i in range(256): #初始化熵计算的临时变量
            self.tmp.append(0)

    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
              (im_height, im_width, 3)).astype(np.uint8)

    def detectBySSD(self,image):

            
            #num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            #print np.array(image).shape
            image_np = np.array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            ( scores, classes) = self.sess.run(
              [ self.detection_scores, self.detection_classes],
              feed_dict={self.image_tensor: image_np_expanded})
            classes=np.squeeze(classes).astype(np.int32)
            scores=np.squeeze(scores)
            return scores,classes

    def getEntropy(self,image):  #计算图像的熵'
        val = 0
        k = 0
        res = 0
        img = np.array(image)
        for i in range(len(img)):
            for j in range(len(img[i])):
                val = img[i][j]
                self.tmp[val] = float(self.tmp[val] + 1)
                k =  float(k + 1)
        for i in range(len(self.tmp)):
            self.tmp[i] = float(self.tmp[i] / k)
        for i in range(len(self.tmp)):
            if(self.tmp[i] == 0):
                res = res
            else:
                res = float(res - self.tmp[i] * (math.log(self.tmp[i]) / math.log(2.0)))
        return res

    def isHuman(self,image):
        scores,classes = self.detectBySSD(image)
        #print classes
        for i in range(classes.size):
            #class=1表示是人类，概率大于设定的阀值
            if (classes[i]==1 and scores[i] > self.human_score_threshold) :
                return True
        return False
    #检测主入口,frame ＰＩＬ.Image类型
    def detect(self,frame):
        image=frame #self.load_image_into_numpy_array(frame)
        fgmask = self.fgbg.apply(image)
        fgmask_entropy= self.getEntropy(fgmask) #图像熵值
        has_human = self.isHuman(image)
        print fgmask_entropy,has_human
        if ( fgmask_entropy > self.entropy_low_threshold and \
            fgmask_entropy<self.entropy_high_threshold and \
            has_human==False):
            return True
        else:
            return False



# In[14]:




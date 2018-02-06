# coding: utf-8


import numpy as np
import tensorflow as tf


class WaterDetector:
    detection_graph = None
    detection_scores = None
    detection_classes = None
    image_tensor = None
    sess = None
    PATH_TO_CKPT = 'faster_rcnn_inception_resnet_v2_atrous_water_2018_02_06/frozen_inference_graph.pb'

    def __init__(self):

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(graph=self.detection_graph, config=config)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

    #  检测主入口,frame, return the score of people detected. return 0 if no people detected.
    def detect(self, frame):
        image_np = np.array(frame)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (scores, classes) = self.sess.run(
            [self.detection_scores, self.detection_classes],
            feed_dict={self.image_tensor: image_np_expanded})
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        # print classes
        for i in range(classes.size):
            # class=1表示是waterloggin，概率大于设定的阀值
            if classes[i] == 1:
                return scores[i]
        return 0

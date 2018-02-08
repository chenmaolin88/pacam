# coding: utf-8

import base64
import threading
import cv2
import motion_detector as md
import object_detector as od
import water_detector  as wd


class BizRuleEngine:
    MSG_NONE = "@@NONE@@"
    MSG_WARNING = "@@WARNING@@"+"可能存在异常"
    MSG_CRITICAL = "@@CRITICAL@@"+"极有可能存在异常"
    entropy_low_threshold = 0.03  # 图像差异熵的最低阀值
    entropy_high_threshold = 1  # 图像差异熵的最高阀值
    human_score_threshold = 0.5  # 只关注概率大于5０％的人类
    water_score_threshold = 0.5  # waterlogging threshold
    bg_reset_interval = 20 # 背景刷新间隔（秒）

    motion_det = None
    object_det = od.ObjectDetector()
    #water_det=  wd.WaterDetector()

    def __init__(self):
        self.motion_det = md.MotionDetector(self.bg_reset_interval)

    def detect_water(self, total_data):
        is_moved = False
        is_human = False
        is_water =False

        imgdata = base64.b64decode(total_data)
        cur_thread = threading.current_thread()
        imgfile = open(cur_thread.name + '.jpg', 'wb')
        imgfile.write(imgdata)
        imgfile.close()

        im = cv2.imread(cur_thread.name + '.jpg')
        fgmask_entropy = self.motion_det.detect(im)
        if self.entropy_high_threshold>fgmask_entropy > self.entropy_low_threshold:
            is_moved = True

        print "is_moved:" + str(is_moved) + ":" + str(fgmask_entropy)

        human_score = BizRuleEngine.object_det.detect(im)

        if human_score > self.human_score_threshold:
            is_human = True
        print "is_human:" + str(is_human) + ":" + str(human_score)

        '''
        water_score = BizRuleEngine.water_det.detect(im)
        if water_score > self.human_score_threshold:
            is_water = True
        print "is_water:" + str(is_water) + ":" + str(water_score)'''

        if is_moved:# and is_water:
            result = BizRuleEngine.MSG_CRITICAL
            '''elif is_water:
                result = BizRuleEngine.MSG_WARNING'''
        else:
            result = BizRuleEngine.MSG_NONE

        if is_human:
            result = BizRuleEngine.MSG_NONE

        return result

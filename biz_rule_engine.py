import base64
import threading
import cv2
import motion_detector as md

class BizRuleEngine:

    MSG_NONE="@@NONE@@"
    MSG_WARNING = "@@WARNING@@"
    MSG_CRITICAL="@@CRITICAL@@"
    entropy_low_threshold =  0.03 #图像差异熵的最低阀值
    entropy_high_threshold =  1 #图像差异熵的最高阀值
    human_score_threshold = 0.1 #只关注概率大于1０％的人类
    motion_det  = None


    def __init__(self):
        self.motion_det = md.MotionDetector()

    def detect_water(self,total_data):
        imgdata = base64.b64decode(total_data)
        cur_thread = threading.current_thread()
        imgfile = open(cur_thread.name + '.jpg', 'wb')
        imgfile.write(imgdata)
        imgfile.close()

        im = cv2.imread(cur_thread.name + '.jpg')
        fgmask_entropy = self.motion_det.detect(im)
        if fgmask_entropy > self.entropy_low_threshold and \
            fgmask_entropy < self.entropy_high_threshold :
            is_moved=True

        if (ret == True):
            result = "@@identify@@"
        else:
            result = "@@@none-identify@"
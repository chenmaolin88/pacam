
# coding: utf-8



import numpy as np
import cv2
import math


class MotionDetector:

    tmp = []  #计算熵值的临时变量
    fgbg = None

    def __init__(self):
        self.fgbg = cv2.BackgroundSubtractorMOG(history=1, nmixtures=5, backgroundRatio=0.1)
        for i in range(256): #初始化熵计算的临时变量
            self.tmp.append(0)

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

    # 检测主入口
    def detect(self,frame):
        image=frame
        fgmask = self.fgbg.apply(image)
        fgmask_entropy= self.getEntropy(fgmask) #图像熵值
        return fgmask_entropy


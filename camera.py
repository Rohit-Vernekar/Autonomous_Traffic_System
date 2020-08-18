import cv2
import numpy as np
import math

def getBoundingBox(contour):
    x = []
    y = []
    for i in contour:
        x.append(i[0])
        y.append(i[1])
    return min(x), min(y), max(x), max(y)


def getSubFrame(contour, img):
    x0, y0, x1, y1 = getBoundingBox(contour)
    subImg = img[y0:y1, x0:x1]
    return subImg


class Camera(object):
    def __init__(self, videoName, vidType, contour):
        self.videoName = videoName
        self.vidType = vidType
        self.contour = contour
        self.video = cv2.VideoCapture('videos/' + self.videoName + '.' + self.vidType)
        self.refImg = cv2.imread('refImg/' + self.videoName + '.png')
        self.avgImg = np.float32(self.refImg)

        self.frame = self.video.read()[1]
        self.mask = np.zeros(self.frame.shape[:2], np.uint8)
        cv2.drawContours(self.mask, [np.array(self.contour).reshape((-1, 1, 2)).astype(np.int32)], -1, (255), -1)
        self.mask = cv2.bitwise_not(self.mask)

    def get_frame(self, generateRefImage):
        ret, frame = self.video.read()
        self.frame = frame.copy()
        if frame is None:
            print("Video Completed")
            return None

        if generateRefImage:
            cv2.accumulateWeighted(frame, self.avgImg, 0.00025, mask=None)
        else:
            cv2.accumulateWeighted(frame, self.avgImg, 0.00025, mask=self.mask)

        self.refImg = cv2.convertScaleAbs(self.avgImg)

        subImage = getSubFrame(self.contour, frame)
        subRefImage = getSubFrame(self.contour, self.refImg)

        pad = 20

        resFrame = cv2.resize(frame, (384, 216))
        resSubImg = cv2.resize(subImage, (384, 216))
        resSubRefImg = cv2.resize(subRefImage, (384, 216))
        padding = np.zeros([216, pad, 3], np.uint8)
        padding = cv2.bitwise_not(padding)

        result = np.zeros((216, 1152 + 4 * pad, 3), np.uint8)
        result[:216, :pad, :3] = padding
        result[:216, pad:pad + 384, :3] = resFrame
        result[:216, pad + 384:384 + 2 * pad, :3] = padding
        result[:216, 384 + 2 * pad:768 + 2 * pad, :3] = resSubImg
        result[:216, 768 + 2 * pad:768 + 3 * pad, :3] = padding
        result[:216, 768 + 3 * pad:1152 + 3 * pad, :3] = resSubRefImg
        result[:216, 1152 + 3 * pad:1152 + 4 * pad, :3] = padding
        cv2.waitKey(10)
        ret, jpeg = cv2.imencode('.jpg', result)
        return jpeg.tobytes()

    def calculateDensity(self):

        newMask = cv2.bitwise_not(self.mask)
        res = cv2.absdiff(self.frame, self.refImg)
        res = cv2.bitwise_and(res, res, mask=newMask)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(res, 0, 255, cv2.THRESH_OTSU)
        if ret < 35:
            ret, thresh = cv2.threshold(res, 35, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=10)

        whitePix = cv2.countNonZero(thresh)
        thresh = cv2.bitwise_not(thresh, mask=newMask)
        blackPix = cv2.countNonZero(thresh)
        density = whitePix * 100 / (whitePix + blackPix)
        return math.ceil(density)
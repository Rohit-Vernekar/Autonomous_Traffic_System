import math

import cv2
import numpy as np

videoName = 'belgaum'
vidType = 'dav'
numberOfSlots = 1


class ROI:
    def __init__(self, img):
        self.contours = []
        self.oimg = img.copy()
        self.dispImg = img.copy()

    def drawROI(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.dispImg = self.oimg.copy()
            self.contours.append((x, y))
            cv2.drawContours(self.dispImg, [np.array(self.contours).reshape((-1, 1, 2)).astype(np.int32)], -1,
                             (255, 255, 255), 10)


def markRoi(img):
    roi = ROI(img)
    cv2.namedWindow("Mark ROI", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Mark ROI", roi.drawROI)
    while True:
        cv2.imshow("Mark ROI", roi.dispImg)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            break
        if k == 27:
            if len(roi.contours) == 0:
                quit(0)
            roi.contours.pop()
            roi.dispImg = roi.oimg.copy()
            if len(roi.contours) == 0:
                roi.dispImg = roi.oimg.copy()
            else:
                cv2.drawContours(roi.dispImg, [np.array(roi.contours).reshape((-1, 1, 2)).astype(np.int32)], -1,
                                 (255, 255, 255), 10)
    cv2.destroyAllWindows()
    return roi.contours


def display(title, img):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)


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


def to3Channel(cImg, gImg):
    res = np.zeros_like(cImg)
    res[:, :, 0] = gImg
    res[:, :, 1] = gImg
    res[:, :, 2] = gImg
    return res


def calculateDensity(frame, refImg, mask, contour, detail):
    mask = cv2.bitwise_not(mask)
    res = cv2.absdiff(frame, refImg)
    abs = getSubFrame(contour, res)

    res = cv2.bitwise_and(res, res, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = getSubFrame(contour, res)

    ret, thresh = cv2.threshold(res, 0, 255, cv2.THRESH_OTSU)
    if ret < 35:
        ret, thresh = cv2.threshold(res, 35, 255, cv2.THRESH_BINARY)
    threshBefore = getSubFrame(contour, thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    threshOpen = getSubFrame(contour, thresh)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=10)
    threshClose = getSubFrame(contour, thresh)

    if detail:
        pad = 20

        cImg = getSubFrame(contour, frame)
        cImg = cv2.resize(cImg, (384, 216))
        abs = cv2.resize(abs, (384, 216))
        gray = cv2.resize(gray, (384, 216))
        gray = to3Channel(cImg, gray)
        threshBefore = cv2.resize(threshBefore, (384, 216))
        threshBefore = to3Channel(cImg, threshBefore)
        threshOpen = cv2.resize(threshOpen, (384, 216))
        threshOpen = to3Channel(cImg, threshOpen)
        threshClose = cv2.resize(threshClose, (384, 216))
        threshClose = to3Channel(cImg, threshClose)

        padding = np.zeros([216, pad, 3], np.uint8)

        result = np.zeros((216, 1920 + 6 * pad, 3), np.uint8)

        result[:216, :pad, :3] = padding
        result[:216, pad:pad + 384, :3] = abs
        result[:216, pad + 384:384 + 2 * pad, :3] = padding
        result[:216, 384 + 2 * pad:768 + 2 * pad, :3] = gray
        result[:216, 768 + 2 * pad:768 + 3 * pad, :3] = padding
        result[:216, 768 + 3 * pad:1152 + 3 * pad, :3] = threshBefore
        result[:216, 1152 + 3 * pad:1152 + 4 * pad, :3] = padding
        result[:216, 1152 + 4 * pad:1536 + 4 * pad, :3] = threshOpen
        result[:216, 1536 + 4 * pad:1536 + 5 * pad, :3] = padding
        result[:216, 1536 + 5 * pad:1920 + 5 * pad, :3] = threshClose
        result[:216, 1920 + 5 * pad:1920 + 6 * pad, :3] = padding

        cv2.imshow("Details", result)

    whitePix = cv2.countNonZero(thresh)
    thresh = cv2.bitwise_not(thresh, mask=mask)
    blackPix = cv2.countNonZero(thresh)
    density = whitePix * 100 / (whitePix + blackPix)
    print("Density : %3d%%" % (math.ceil(density)))
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video = cv2.VideoCapture('videos/' + videoName + '.' + vidType)
    ret, frame = video.read()

    if numberOfSlots == 1:
        generateRefImage = False
        detail = False
        refImg = cv2.imread('refImg/' + videoName + '.png')
        # refImg = frame.copy()

        contour = markRoi(frame)
        # print(contour)
        # contour = [(1166, 480), (1531, 268), (1680, 268), (1684, 552)]
        avgImg = np.float32(refImg)

        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(mask, [np.array(contour).reshape((-1, 1, 2)).astype(np.int32)], -1, (255), -1)
        mask = cv2.bitwise_not(mask)

        while True:
            ret, frame = video.read()

            if frame is None:
                print("Video Completed")
                break

            if generateRefImage:
                cv2.accumulateWeighted(frame, avgImg, 0.00025, mask=None)
            else:
                cv2.accumulateWeighted(frame, avgImg, 0.00025, mask=mask)

            refImg = cv2.convertScaleAbs(avgImg)
            subImage = getSubFrame(contour, frame)
            subRefImage = getSubFrame(contour, refImg)

            pad = 20

            resFrame = cv2.resize(frame, (384, 216))
            resSubImg = cv2.resize(subImage, (384, 216))
            resSubRefImg = cv2.resize(subRefImage, (384, 216))
            padding = np.zeros([216, pad, 3], np.uint8)

            result = np.zeros((216, 1152 + 4 * pad, 3), np.uint8)
            result[:216, :pad, :3] = padding
            result[:216, pad:pad + 384, :3] = resFrame
            result[:216, pad + 384:384 + 2 * pad, :3] = padding
            result[:216, 384 + 2 * pad:768 + 2 * pad, :3] = resSubImg
            result[:216, 768 + 2 * pad:768 + 3 * pad, :3] = padding
            result[:216, 768 + 3 * pad:1152 + 3 * pad, :3] = resSubRefImg
            result[:216, 1152 + 3 * pad:1152 + 4 * pad, :3] = padding

            cv2.imshow("Result", result)

            k = cv2.waitKey(20)
            if k == ord('c'):
                calculateDensity(frame, refImg, mask, contour, detail)
                print("\n\n\n")
            elif k == ord('p'):
                print("Video Paused")
                while cv2.waitKey() != ord('p'):
                    continue
                print("Video Resumed")
            elif k == ord('d'):
                detail = not detail
                print("Detail : ", detail)
            elif k == ord('s'):
                generateRefImage = not generateRefImage
                print("Start : ", generateRefImage)
            elif k == 27:
                cv2.destroyAllWindows()
                ans = input("Save Reference Image? (y/n) : ")
                if ans == 'y' or ans == 'Y':
                    cv2.imwrite('refImg/' + videoName + '.png', refImg)
                    print("Reference Image saved Successfully.")
                else:
                    print("Reference Image not saved.")
                break

    video.release()

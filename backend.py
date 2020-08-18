import cv2




import math
import numpy as np
from flask import Flask, render_template, Response, jsonify

from camera import Camera

contour = None
videoName = 'hubli1'
vidType = 'MOV'
generateRefImage = False
pauseVideo = False
oldFrame = None
cameraObj = None
app = Flask(__name__, static_folder="templates/static")


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


def gen(camera):
    global oldFrame
    while True:
        if pauseVideo:
            frame = oldFrame
        else:
            frame = camera.get_frame(generateRefImage)
            oldFrame = frame
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    global cameraObj
    return Response(gen(cameraObj), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/getDensity', methods=['GET'])
def getDensity():
    global cameraObj
    density = cameraObj.calculateDensity()
    timer = math.ceil(density * 40 / 100)
    return jsonify(density=density, timer=timer)


@app.route('/startVideo', methods=['GET'])
def start():
    global generateRefImage
    generateRefImage = True
    print("Start")
    return jsonify(start=1)


@app.route('/stopVideo', methods=['GET'])
def stop():
    global generateRefImage
    generateRefImage = False
    print("Stop")
    return jsonify(stop=1)


@app.route('/pauseVideo', methods=['GET'])
def pause():
    global pauseVideo
    if pauseVideo:
        print("Play")
        pauseVideo = False
        return jsonify(pause=0)
    else:
        print("Pause")
        pauseVideo = True
        return jsonify(pause=1)


if __name__ == '__main__':
    video = cv2.VideoCapture('videos/' + videoName + '.' + vidType)
    ret, oldFrame = video.read()
    contour = markRoi(oldFrame)

    cameraObj = Camera(videoName, vidType, contour)

    app.run(debug=True, use_reloader=False)

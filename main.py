import time

import cv2
import imutils
import numpy as np
import tensorflow as tf
import tflearn
from PIL import Image
from pynput.keyboard import Key, Controller
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# global variables
bg = None
keyboard = Controller()
is_stopped = True


def resize_image(image_name):
    base_width = 100
    img = Image.open(image_name)
    width_percent = (base_width / float(img.size[0]))
    height_size = int((float(img.size[1]) * float(width_percent)))
    img = img.resize((base_width, height_size), Image.ANTIALIAS)
    img.save(image_name)


def run_avg(image, average_weight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, average_weight)


def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    img_threshold = cv2.threshold(diff,
                                  threshold,
                                  255,
                                  cv2.THRESH_BINARY)[1]

    # get the contours in the threshold image
    (contours, _) = cv2.findContours(img_threshold.copy(),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(contours) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(contours, key=cv2.contourArea)
        return img_threshold, segmented


def main():
    # initialize weight for running average
    average_weight = 0.5

    # get the reference to the web camera
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    start_recording = False

    # keep looping, until interrupted
    while True:
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to gray scale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, average_weight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the threshold image and
                # segmented region
                (threshold, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    cv2.imwrite('Temp.png', threshold)
                    resize_image('Temp.png')
                    predicted_class, confidence = get_predicted_class()
                    show_statistics(predicted_class, confidence)
                cv2.imshow("Thresholded", threshold)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

        if keypress == ord("s"):
            start_recording = True


def get_predicted_class():
    # Predict
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))


def show_statistics(predicted_class, confidence):
    global is_stopped
    text_image = np.zeros((300, 512, 3), np.uint8)
    class_name = ""
    if predicted_class == 0:
        class_name = "Swing"
        if confidence >= 0.99:
            print("Detected: " + class_name)
        # if isStoped == False:
        #     isStoped = True
        #     print("Stopping")
        #     time.sleep(0.5)
        #     keyboard.press(Key.space)
        #     keyboard.release(Key.space)
    elif predicted_class == 1:
        class_name = "Palm"
        if confidence >= 0.99:
            print("Detected: " + class_name)
            if is_stopped:
                is_stopped = False
                print("Starting")
                time.sleep(0.5)
                keyboard.press(Key.space)
                keyboard.release(Key.space)
    elif predicted_class == 2:
        class_name = "Fist"
        if confidence >= 0.9999:
            print("Detected: " + class_name)
            if not is_stopped:
                is_stopped = True
                print("Stopping")
                time.sleep(0.5)
                keyboard.press(Key.space)
                keyboard.release(Key.space)

    cv2.putText(text_image, "Predicted Class : " + class_name,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    cv2.putText(text_image, "Confidence : " + str(confidence * 100) + '%',
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
    cv2.imshow("Statistics", text_image)


# Model defined
tf.reset_default_graph()
convnet = input_data(shape=[None, 89, 100, 1], name='input')
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1000, activation='relu')
convnet = dropout(convnet, 0.75)

convnet = fully_connected(convnet, 3, activation='softmax')

convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='regression')

model = tflearn.DNN(convnet, tensorboard_verbose=0)

# Load Saved Model
model.load("TrainedModel/GestureRecogModel.tfl")

main()

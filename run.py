import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import model
import cv2
from subprocess import call
import os
from send_steering_angle import send_angle  # Import the function to send angle to Arduino

#check if on windows OS
windows = False
if os.name == 'nt':
    windows = True

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

i = 1
while(cv2.waitKey(70) != ord('q')):
    full_image = cv2.imread("self-driving-car-dataset-master/self-driving-car-dataset-master/IMG/" + str(i) + ".jpg")
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / 3.14159265
    if not windows:
        call("clear")
    print("Predicted steering angle: " + str(degrees) + " degrees")
    send_angle(degrees)  # Send the predicted angle to Arduino
    cv2.imshow("frame", full_image)
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    # Fixed division by zero bug - handle when degrees equals smoothed_angle
    diff = degrees - smoothed_angle
    if abs(diff) > 0.0001:  # Avoid division by zero
        sign = 1 if diff > 0 else -1
        smoothed_angle += 0.2 * pow(abs(diff), 2.0 / 3.0) * sign
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()

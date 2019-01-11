# DROWSY-DRIVER

This is an Attention Score based algorithm based on Blinking Rate and Yawning of the  Driver, based on which it warns him as well as takes the control of the vehicle if the Attention score fall below certain level. This is acheived on hardware level by using  CAN protocol to control Vehicle autonomously. 

Eye state classification using OpenCV and DLib to estimate Percentage Eye Closure as well Yawn detection using DLib and OpenCV.

Uses DLib facial landmark detector to find the major and minor axes of eyes, as well as mouth. The aspect ratio of major to minor axes is used to determine whether eye/mouth is open; which allows for eye-state classification and yawning detection. Requires a pre-trained DLib facial landmark detector model in a .dat file.

# Pseudo Code
```
Pseudo Code : 

Attention Score determining Algorithm (assuming 30fps):

Assign Initial Positive Attention Max. Score : 80000
Define Warning Level : 8000
Define AutoBrake Level : 4000

checkWarning : 
	if Attention Score < 8000
		Display WARNING
	if Attention Score < 4000
		Take Control and Slow Down Vehicle


If Yawn and Blink is not detected in a Frame
	Attention Score += 60

If Blink is Detected
	Attention Score -= 20 x Frame Counter
If Yawn is Detected in a frame
	Attention Score -= 2 x Frame Counter for Yawn
	checkWarning()
_______________________________________________________
_______________________________________________________

Code Flow ->

read frame from camera
convert into grayscale 
identify face landmark using dlib

yawn detection:
	distance between lips > yawning threshold
		Frame Counter (with Yawn Detected) ++
		if Yawning in Continuos Number of Frames > Threshold
			Attention Score -=2x Frame Counter
			checkWarning // check warning status after each score update
If Yawn and Blink is not detected in a Frame
	Attention Score += 60 (if Attention Score < Fixed Maximum Score)
	checkWarning // check warning status after each score update

Eye Blinking detection:
	Using PERCLOS
	Eyes aspect ratio < Average aspect ratio (parameter)
	Blink is detected
		if Bink is detected continuously for Threshold no. of frames
			Drowsiness Detected
			Attention Score -=20 x Frame Counter
			checkWarning // check warning status after each score update
```

# Setup and Dependecies


## dlib library [![Travis Status](https://travis-ci.org/davisking/dlib.svg?branch=master)](https://travis-ci.org/davisking/dlib)

Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. See [http://dlib.net](http://dlib.net) for the main project documentation and API reference.

We are using the python for coding on **PYNQ** board. Hence dlib's python API is installed.


## Compiling dlib Python API

Before you can run the Python example programs you must compile dlib. Type:

```bash
python setup.py install
```

## Controller Area Network(CAN) Protocol 

### Now we are fully equiped to understand and implement our code.

## Explanation of code module-wise

```
import cv2
import dlib
import time
import imutils
import argparse
import numpy as np
from threading import Thread
from collections import OrderedDict
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from scipy.spatial import distance as dist

SHAPE_PREDICTOR_PATH = "/home/rey/shape_predictor_68_face_landmarks.dat"
```
Files are imported into the program and path for trained wights is provided.

```
def eye_aspect_ratio(eye):

    vertical_A = dist.euclidean(eye[1], eye[5])
    vertical_B = dist.euclidean(eye[2], eye[4])
    horizontal_C = dist.euclidean(eye[0], eye[3])
    ear = (vertical_A + vertical_B) / (2.0 * horizontal_C)
    return ear
```
The function returns euclidean distances between the two sets of vertical and horizontal eye landmarks in cartesian coordinates. Next it computes the eye aspect ratio and return the same.

```
def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
```
This funtion generate landmarks on the face by using a pre-trained model predictor

```
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im
```
Copy the landmarks on the image frame and return it.
```
def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])   
```
Compute and put landmarks on the top lip on the image frame.
```
def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])
```
Compute and put landmarks on the bottom lip on the image frame.

```
def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance
```    
Find the absolute distance between the top and bottom lips using the topmost landmark of top lip and bottomost landmark of bottom lip.

```
# Constant for the eye aspect ratio to indicate drowsiness 
EYE_AR_THRESH = 0.25
# Constant for the number of consecutive frames the eye (closed) must be below the threshold
EYE_AR_CONSEC_FRAMES = 10
# Initialize the frame counter
FRAME_COUNTER = 0
# Boolean to indicate if the alarm is going off
IS_ALARM_ON = False
#yawning distance threshold 
YAWN_DIST = 26
#Maximum Positive Attention Score : 
AttentionScoreMax = 80000
AttentionScore = AttentionScoreMax
#Warning Level
WarningLevel = 40000
autoBrakeLevel = 6000
#error frame 
error_frame_thres = 2
YAWN_MIN_FRAME_COUNT = 10
```
All the threshold values and variables are initialized.

```
def warningAlert():
    print (" WARNING ALERT !!! ")
    output_text = " Attention Score " + str(AttentionScore)
    cv2.putText(frame,output_text,(30,300),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
    return
```
This function gives an alert on the screen and outputs the attention score of the driver.

```
def autoBrakeLevel():
    print (" AUTO BRAKE !!! ")
    return
```
This function when called executes the CAN protocol for hardware level implementations 
on the car for slow braking and alerting the driver about it.
```  
def checkWarning():
    if(AttentionScore < WarningLevel):
        warningAlert()
    if(AttentionScore < autoBrakeLevel):
        autoBrakeLevel()
```
Compare the attention score and threshold attention level.

```
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)
```
Take a bounding predicted by dlib and convert it to the format (x, y, w, h) as normally handled in OpenCV.

```
def shape_to_np(shape, dtype = 'int'):
    coords = np.zeros((68, 2), dtype = dtype)
    
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        
    return coords
```
The dlib face landmark detector will return a shape object containing 
the 68 (x, y)-coordinates of the facial landmark regions. This fucntion converts the above object to a NumPy array.

```
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])
```
This function define a dictionary that maps the indexes of the facial landmarks to specific face regions.

```
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
(leStart, leEnd) = FACIAL_LANDMARKS_IDXS['left_eye']
(reStart, reEnd) = FACIAL_LANDMARKS_IDXS['right_eye']
vs = VideoStream(src = 0).start()
fileStream = False
time.sleep(1.0)
yawns = 0
yawn_status = False 
FRAME_COUNTER_YAWN =0
FRAME_COUNTER_EYES =0
error_frame =0
yawns = 0
```
Initialize dlib's face detector (HOG-based), creates a facial landmark predictor, get the stream from a camera and define all variables.

```
while True:
    global frame 
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    image_landmarks, lip_distance = mouth_open(frame)
    prev_yawn_status = yawn_status  
    print ("lip distance: ", lip_distance)
    print ("AttentionScore: ", AttentionScore)
    if (lip_distance > YAWN_DIST):
        FRAME_COUNTER_YAWN +=1
        yawn_status == True
        print ("Frame Count: ", FRAME_COUNTER)
        if(FRAME_COUNTER_YAWN > YAWN_MIN_FRAME_COUNT): 
                AttentionScore -=2*FRAME_COUNTER
                cv2.putText(frame, "Subject is Yawning",(50,450),cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
                checkWarning()
            
    else:
        yawn_status == False
        if (AttentionScore < AttentionScoreMax):
            AttentionScore +=10
            checkWarning()
            
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1
        FRAME_COUNTER_YAWN = 0
```
Master code of this program - It gets the yawn status and counts the number of frames for a yawning period. It then ammends the attention score of the driver based on the frame count which is directly related to the drowsiness of the driver. Also it keeps a account of no. of yawns by a person.

```
    cv2.imshow('Live Landmarks', image_landmarks )
    cv2.imshow('Yawn Detection', frame )
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape_np = shape_to_np(shape)
        
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape_np[leStart:leEnd]
        rightEye = shape_np[reStart:reEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # average the eye aspect ratio together for both eyes
        avgEAR = (leftEAR + rightEAR) / 2.0
        
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if avgEAR < EYE_AR_THRESH:
            FRAME_COUNTER_EYES += 1
            
            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if FRAME_COUNTER_EYES >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, 'DROWSINESS ALERT!!!', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                AttentionScore-=20*FRAME_COUNTER_EYES
                checkWarning()
            
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        else:
            FRAME_COUNTER_EYES = 0
            IS_ALARM_ON = False
```
This fragment of the code computes the area of eyes and decides the drowsiness of the driver. It also ammends the attention score but with a very low weight as compared to that of a yawn. It checks if the eyes are closed for sufficient amount of time as well.

```
        # draw the computed eye aspect ratio for the frame
        cv2.putText(frame, "EAR: {:.2f}".format(avgEAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        output_text = " Attention Score " + str(AttentionScore)
        cv2.putText(frame,output_text,(30,300),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        #cv2.putText(frame, output_text, (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
```
This is the last part of the code which combines all the above modeules into one program and displays the live result on the screen.


============================================

The Project has Two Models : 

Model 1 :  (ALERT/WARNING Signals Part can be implemented by *anyone on any Car with Music System*)

    Attention Score is determined on the basis of alertness judged by blinking eyes pattern and yawing rate. 
    WARNING SIGNALS (ALERT SOUND and HAZARD INDICATORS) are activated when Attention Score falls before a certain WARNING LEVEL.
    If Driver is still remains asleep,Attention score keeps falling and our [Autonomous Braking Algorithm](doc.link) is deployed to stop the vehicle and minimize the damage.

Model 2:    (This Requires the Car to have [CAN-bus Interface](doc.here) for Autonomous Braking to work and Odometry sensors to return velocity)

    It includes several improvements over first model.
    *   Attention Score Penality rate takes **vehicle's velocity factor** into consideration, as the Attention Penality for closing eyes/yawning at high speed must be greater than that in low speed because of the higher level of required altertness at high speed. Example: Closing Eyes for 1sec at High Speed is much more significant than closing it at very low speed.
    [ATTENTION SCORE ALGORITHM : MODEL 2](doc.here)
    *   Improvised Braking Algorithm including factors such as Traffic and nearest object distance into cosideration with velocity of car, braking distance (including reaction time of driver and alertness level).
    [BRAKING ALGORITHM : MODEL 2](doc.here)

============================================




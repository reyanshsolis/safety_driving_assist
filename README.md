# drowsy_driver
This keeps maintains an Attention Score based on Blinking Rate and Yawning of Driver, based on which it Warns him takes control of the vehicle if Attention score fall below certain level using CAN protocol to control Vehicle autonomously. 


#DLIB


# dlib C++ library [![Travis Status](https://travis-ci.org/davisking/dlib.svg?branch=master)](https://travis-ci.org/davisking/dlib)

Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. See [http://dlib.net](http://dlib.net) for the main project documentation and API reference.

We are using the python for coding on **PYNQ** board. Hence dlib's python API is installed.


## Compiling dlib Python API

Before you can run the Python example programs you must compile dlib. Type:

```bash
python setup.py install
```


# CAN Protocol 
###  KVASER installation [Link](https://www.kvaser.com/linux-drivers-and-sdk/)

### CANLib installation [Link](https://www.youtube.com/watch?v=Gz-lIVIU7ys&feature=youtu.be)
 

#### Now we are fully equiped to understand and implement our code.

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
Files are imported into the program and path for trained wights is provided

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


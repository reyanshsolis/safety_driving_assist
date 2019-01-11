# DROWSY-DRIVER

This is an Attention Score based algorithm based on Blinking Rate and Yawning of the  Driver, based on which it warns him as well as takes the control of the vehicle if the Attention score fall below certain level. This is acheived on hardware level by using  CAN protocol to control Vehicle autonomously. 

Eye state classification using OpenCV and DLib to estimate Percentage Eye Closure as well Yawn detection using DLib and OpenCV.

Uses DLib facial landmark detector to find the major and minor axes of eyes, as well as mouth. The aspect ratio of major to minor axes is used to determine whether eye/mouth is open; which allows for eye-state classification and yawning detection. Requires a pre-trained DLib facial landmark detector model in a .dat file.


### The Project has Two Models : 

## Model 1 :  
(WARNING Signals Part can be implemented by *anyone on any Car with Music System*) (Automous Braking can be implemented on any Bot, however it requires CAN-bus interface for a real Car)
*   Attention Score is determined on the basis of alertness judged by blinking eyes pattern and yawing rate. 
WARNING SIGNALS (ALERT SOUND and HAZARD INDICATORS) are activated when Attention Score falls before a certain WARNING LEVEL.
If Driver is still remains asleep,Attention score keeps falling and our [Autonomous Braking Algorithm](doc.link) is deployed to stop the vehicle and minimize the damage.

## Model 2:    (This Requires the Car to have [CAN-bus Interface](doc.here) for Autonomous Braking to work and Odometry sensors to return velocity)
It includes several improvements over first model.
*   Attention Score Penality rate takes **vehicle's velocity factor** into consideration, as the Attention Penality for closing eyes/yawning at high speed must be greater than that in low speed because of the higher level of required altertness at high speed. Example: Closing Eyes for 1sec at High Speed is much more significant than closing it at very low speed.
[ATTENTION SCORE ALGORITHM : MODEL 2](doc.here)

*   Improvised Braking Algorithm including factors such as Traffic and nearest object distance into cosideration with velocity of car, braking distance (including reaction time of driver and alertness level).
[BRAKING ALGORITHM : MODEL 2](doc.here)


# Pseudo Code
## MODEL 1 : 
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
## [INSTALLATION AND GETTING STARTED GUIDE](https://github.com/reyanshsolis/safety_driving_assist/blob/master/Setup_Getting_Started.md)

## Detailed Explanation of entire Code: [Detailed_Code_Explanation.md](https://github.com/reyanshsolis/safety_driving_assist/blob/master/Detailed_Code_Explanation.md)

============================================
============================================




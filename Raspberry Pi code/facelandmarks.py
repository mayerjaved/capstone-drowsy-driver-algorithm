import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
import numpy as np
from playsound import playsound
import RPi.GPIO as GPIO
import time



#these variables are used to adjust the eye closed threshold, higher number is more sensitive, lower number is less sensitive
eyeDistcheck = 10
headAnglecheck = 7
irisDistcheck = 7
irisDistcheckLR = 9
irisAngleBottom = 168
DrowsyDriver = False
sound_playing = False
message1 = ''
message2 = ''
message3 = ''
distThresX = 150
distThresY = 100

#the following code is extracted from
#https://www.samproell.io/posts/yarppg/yarppg-face-detection-with-mediapipe/
#the following function takes the captured face mesh landmark coordinates and 
#turns them into a list of tuples representing the (x,y) positions on the image
def getFacemeshCoords(img, landmark_list):
    """Extract FaceMesh landmark coordinates into 468x2 NumPy array.
    """
    h, w = img.shape[:2]  # grab width and height from image
    xy = [(lm.x, lm.y) for lm in landmark_list.landmark]    
    #we multiply x*w and y*h to scale the landmark co-ordinates range of [0, 1] 
    # to the actual pixel coordinates in the image by multiplying them by width and height
    return [(int(x * w), int(y * h)) for x, y in xy]


#This function is used to print the mesh points on the camera, mostly 
# intended for testing purposes as it slows the program down
def printMesh(image, mesh_points):
  for i, (x, y) in enumerate(mesh_points):
    cv2.circle(image, (x, y), 2, (255, 255, 0), -1)
    cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


#this function compares the cordinates of the center of the image and the center of the face to make sure the driver is in the frame
#it also checks the angle of the face to make sure its facing the camera and not at an extreme angle
#since this function captures landmarks and does calculations required to calculate the head pitch angle I built that feature into this function
#instead or dedicating a seperate function for that feature
def faceInCenter(image, mesh_points):
  global message1
  imgY, imgX, _ = image.shape
  imgXmid = imgX/2
  imgYmid = imgY/2
  nosepoint = mesh_points[4]
  rightEyeRight =  mesh_points[446]
  leftEyeLeft = mesh_points[130]
  bottomLip = mesh_points[18]
  forehead = mesh_points[151]
  faceCenterX =  (nosepoint[0]+bottomLip[0]+forehead[0])/3
  faceCenterY = (nosepoint[1]+bottomLip[1]+forehead[1])/3
  distanceFromX = abs(faceCenterX-imgXmid)
  distanceFromY = abs(faceCenterY-imgYmid)

  #this part of the function finds the angle between the nose and the 2 eyes and determines the head pitch angle accordingly
  distC = distanceCalculator(rightEyeRight,leftEyeLeft)
  #cv2.line(image, rightEyeRight, leftEyeLeft, (0, 255, 0), thickness=2)
  distA = distanceCalculator(rightEyeRight, nosepoint)
  #cv2.line(image, rightEyeRight, nosepoint, (0, 255, 0), thickness=2)
  distB = distanceCalculator(leftEyeLeft, nosepoint)
  #cv2.line(image, leftEyeLeft, nosepoint, (0, 255, 0), thickness=2)
  #angleA = ((distA**2)+(distB**2)-(distC**2))/(2*distA*distB)


  #finds the angle
  angleA = ((distC**2)+(distB**2)-(distA**2))/(2*distC*distB)
  angleB = ((distC**2)+(distA**2)-(distB**2))/(2*distC*distA)
  angleC = ((distA**2)+(distB**2)-(distC**2))/(2*distA*distB)

  #runs cos inverse, converts the value to degree and sets 2 decimal places
  angleA = round(angleA * (180 / 3.14),1)
  angleB = round(angleB * (180 / 3.14),1)
  angleC = (round(angleC * (180 / 3.14),1))
  #print(angleA, ' ',angleB, ' ', angleC)

  
  #this condition will take place when the user's face not in the center of the camera or the face is tilted significantly left or right
  if (distanceFromX > distThresX and distanceFromY > distThresY) or (abs(angleA - angleB) > 40):
    message1 = "Please make sure to directly face the camera to continue"
    #cv2.putText(image, message1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, message1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    rect_size = 600
    rect_left = int((imgX - rect_size) / 2)
    rect_top = int((imgY - rect_size) / 2)
    rect_right = rect_left + rect_size
    rect_bottom = rect_top + rect_size
    cv2.rectangle(image, (rect_left, rect_top), (rect_right, rect_bottom), (255, 0, 0), 2)
  # draw the rectangle
    return False
  elif angleC > headAnglecheck and (abs(angleA - angleB) < 40):
    message1 = "Head angle: drowsy "
    
  else:
    message1 = "Head angle: awake "
    
  #message1 += "bottom: " + str(angleC) + " Left:"+ str(angleA) + " Right: " + str(angleB)
  cv2.putText(image, message1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  return True
  
  
  
   
#this function measures the distance between the top of the eye lid and the bottom of the 
# eye lid and determines if the eyes are closed 
def eyesClosed(image, mesh_points, countLagEyes):
  global message2
  #status is a bool variable which will keep track of if the eyes are open or close and then the funtion returns this bool variable for other functions to know if the eyes are closed or not
  status = False
  rightTop = mesh_points[159]
  rightBottom = mesh_points[145]
  distRight = distanceCalculator(rightTop,rightBottom)

  leftTop = mesh_points[386]
  leftBottom = mesh_points[374]
  distLeft = distanceCalculator(leftTop,leftBottom)

  distRatio = (distRight + distLeft)/2

  if (distRatio < eyeDistcheck): 
    status = True
    countLagEyes = countLagEyes + 1
    if countLagEyes >= 7:
      print(countLagEyes)
      DrowsyDriver = True
      playSound(DrowsyDriver)
      message2 = "Eyes: closed"
  else:
      DrowsyDriver = False
      message2 = "Eyes: open"   
      playSound(DrowsyDriver)
      countLagEyes = 0
  cv2.putText(image, message2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  return countLagEyes

def eyeIris(image,mesh_points, countLag):
  rightIrisLeft = mesh_points[471]
  rightIrisMid = mesh_points[468]
  rightIrisRight = mesh_points[469]
  rightRight1 = mesh_points[173] #173 innter or 133
  rightLeft1 = mesh_points[130] #33 inner or 130 outer

  rightRight = mesh_points[190] #173 innter or 133 or 190 or 189
  rightLeft = mesh_points[130] #33 inner or 130 outer
  distTotal1 = distanceCalculator(rightRight,rightLeft)
  #cv2.line(image, rightRight,rightLeft, (0, 255, 0), thickness=2)
  distLeft1 = distanceCalculator(rightRight, rightIrisMid)
  #cv2.line(image, rightRight, rightIrisMid, (0, 255, 0), thickness=2)
  distRight1 = distanceCalculator(rightIrisMid, rightLeft)
  #cv2.line(image, rightIrisMid, rightLeft, (0, 255, 0), thickness=2)

  angleRight = ((distLeft1**2)+(distRight1**2)-(distTotal1**2))/(2*distLeft1*distRight1)
  angleRight = round(angleRight * (180 / 3.14),1)

  leftIrisLeft = mesh_points[476]
  leftIrisRight = mesh_points[474]
  leftIrisMid = mesh_points[473]
  
 
  leftRight = mesh_points[263] #466 or 263
  leftLeft = mesh_points[414] #398 or 414 or 413
  distTotal = distanceCalculator(leftRight,leftLeft)
  #cv2.line(image, leftRight,leftLeft, (0, 255, 0), thickness=2)
  distLeft = distanceCalculator(leftRight, leftIrisMid)
  #cv2.line(image, leftRight, leftIrisMid, (0, 255, 0), thickness=2)
  distRight = distanceCalculator(leftIrisMid, leftLeft)
  #cv2.line(image, leftIrisMid, leftLeft, (0, 255, 0), thickness=2)


  #finds the angle
  angleLeft = (((distLeft**2)+(distRight**2))-(distTotal**2))/(2*distLeft*distRight)
  angleLeft = round(angleLeft * (180 / 3.14),1)


  #making sure that the midIris is below the left and right part of the eye, which means that the user is looking down
  if ((leftRight[1]+leftLeft[1])/2) < leftIrisMid[1] and angleLeft < irisAngleBottom and ((rightRight[1]+rightLeft[1])/2) < rightIrisMid[1] and angleRight < irisAngleBottom:
    message3 = "looking down "
    #countLag is a variable used to create a bit of a time to make sure that the user has been looking down for a second before the sound is played 
    countLag = countLag + 1
    if countLag >= 10:
      DrowsyDriver = True
      playSound(DrowsyDriver)
  elif ((leftRight[1]+leftLeft[1])/2) > leftIrisMid[1] and angleLeft < irisAngleBottom and ((rightRight[1]+rightLeft[1])/2) > rightIrisMid[1] and angleRight < irisAngleBottom:
    message3 = "looking up "
    countLag = 0

    #countLag is a variable used to create a bit of a time to make sure that the user has been looking down for a second before the sound is played 
  #if the mid iris point is above the eye level it indicates that the user is looking up 
  elif distanceCalculator(rightIrisLeft, rightLeft1) < irisDistcheckLR and distanceCalculator(leftIrisLeft, leftLeft) < irisDistcheckLR:
    message3 = "looking left"
    countLag = 0

  elif distanceCalculator(rightIrisRight, rightRight1) < irisDistcheckLR and distanceCalculator(leftIrisRight, leftRight) < irisDistcheckLR:
    message3 = "looking right"
    countLag = 0

  else:
    message3 = "looking straight "
    countLag = 0
    DrowsyDriver = False
    playSound(DrowsyDriver)

   

  #left
  #cv2.line(image, rightIrisLeft, rightLeft1, (0, 255, 0), thickness=2)
  #cv2.line(image, leftIrisLeft, leftLeft, (0, 255, 0), thickness=2)
  #right
  #cv2.line(image, leftIrisRight, leftRight, (0, 255, 0), thickness=2)
  #cv2.line(image, rightIrisRight, rightRight, (0, 255, 0), thickness=2)
  #Showing the angles on screen for testing
  #message3 += "left: " + str(angleLeft) + " right: " + str(angleRight) 
  cv2.putText(image, message3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  return countLag



#this function takes the x and y cordinates of 2 points and calculates the distance between them
def distanceCalculator(point1, point2):
  return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5



#The playSound() function passed a 3.3v pulse to the #23 GPIO pin basically triggering 
# the 2n7000 mosfet and turning on the analog speaker
def playSound(DrowsyDriver):
    pin_number = 23
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin_number, GPIO.OUT)
    if DrowsyDriver:
        GPIO.output(pin_number, GPIO.LOW)
        print('playing sound')
    else:
        GPIO.output(pin_number, GPIO.HIGH)
        print('sound off')
    #GPIO.cleanup()


#The steeringCheck() function continuously reads the input of thn GPIO pin 16. 
# This pin is in the pull up resistor mode and if pulsed to ground is know that the user is holding the steering wheel
def steeringCheck(image):
    GPIO.setmode(GPIO.BCM)
    pin_number = 16
    GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    input_value = GPIO.input(16)
    # Print the value
    if not  input_value:
        cv2.putText(image, 'Holding steering :)', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(image, 'Not holding steering :(', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    GPIO.cleanup()




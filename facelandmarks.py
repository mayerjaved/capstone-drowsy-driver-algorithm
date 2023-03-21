import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
import numpy as np
from playsound import playsound
import RPi.GPIO as GPIO


#these variables are used to adjust the eye closed threshold, higher number is more sensitive, lower number is less sensitive
eyeDistcheck = 10
headAnglecheck = 90
irisDistcheck = 7
irisDistcheckLR = 11
DrowsyDriver = False
sound_playing = False


#the following code is extracted from
#https://www.samproell.io/posts/yarppg/yarppg-face-detection-with-mediapipe/
#the following function takes the captured face mesh landmark coordinates and 
#turns them into a list of tuples representing the (x,y) positions on the image
def get_facemesh_coords(img, landmark_list):
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


#this function measures the distance between the top of the eye lid and the bottom of the 
# eye lid and determines if the eyes are closed 
def eyesClosed(image, mesh_points):
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
    message = "Eyes: closed"
  else:
    status = False
    message = "Eyes: open"    
  cv2.putText(image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


#this part of the function finds the angle between the nose and the 2 eyes and determines the head pitch angle accordingly
  nosepoint = mesh_points[4]
  rightEyeRight =  mesh_points[446]
  leftEyeLeft = mesh_points[130]

  distC = distanceCalculator(rightEyeRight,leftEyeLeft)
  #cv2.line(image, rightEyeRight, leftEyeLeft, (0, 255, 0), thickness=2)
  distA = distanceCalculator(rightEyeRight, nosepoint)
  #cv2.line(image, rightEyeRight, nosepoint, (0, 255, 0), thickness=2)
  distB = distanceCalculator(leftEyeLeft, nosepoint)
  #cv2.line(image, leftEyeLeft, nosepoint, (0, 255, 0), thickness=2)
  angleA = ((distA**2)+(distB**2)-(distC**2))/(2*distA*distB)


  #finds the angle
  angleA = ((distC**2)+(distB**2)-(distA**2))/(2*distC*distB)
  angleB = ((distC**2)+(distA**2)-(distB**2))/(2*distC*distA)
  angleC = ((distA**2)+(distB**2)-(distC**2))/(2*distA*distB)


  #runs cos inverse, converts the value to degree and sets 2 decimal places
  angleA = round(math.degrees(math.acos(angleA)) , 1)
  angleB = round(math.degrees(math.acos(angleB)) , 1)
  angleC = round(math.degrees(math.acos(angleC)) , 1)

  
  if angleC < headAnglecheck and (abs(angleA - angleB) < 40):
    message = "Angle: drowsy "
    if status:
       DrowsyDriver = True
       play_sound(DrowsyDriver)  

  else:
    message = "Angle: awake "
    DrowsyDriver = False
    play_sound(DrowsyDriver)  


  #message += "bottom: " + str(angleC) + " Left:"+ str(angleA) + " Right: " + str(angleB)
  cv2.putText(image, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


#this function will estimate the position of the user's iris position to determine wether they are
#looking stright ahead or no
def eyeIris(image,mesh_points, countLag):
  
  rightIrisMid = mesh_points[468]
  rightIrisLeft = mesh_points[471]
  rightIrisRight = mesh_points[469]

  rightTopRight = mesh_points[157]
  rightTopRight1 = mesh_points[158]
  rightTopLeft = mesh_points[160]
  rightTopLeft1 = mesh_points[161]

  rightRight = mesh_points[173] #or 133
  rightLeft = mesh_points[33] #or 130
  rightBottom = mesh_points[145]
  rightTop = mesh_points[159]

  rightBottomLeft = mesh_points[163]
  rightBottomLeft1 =  mesh_points[144]
  rightBottomRight = mesh_points[153]
  rightBottomRight = mesh_points[154]

  leftIrisMid = mesh_points[473]
  leftIrisLeft = mesh_points[476]
  leftIrisRight = mesh_points[474]

  leftRight = mesh_points[466] #466 or 263
  leftLeft = mesh_points[398] #or 414
  leftBottom = mesh_points[374]
  leftTop = mesh_points[386]


  #we have a few line drawings for testing to make sure the right points from the 
  #top
  #cv2.line(image, rightIrisMid, rightTop, (0, 255, 0), thickness=2)
  #cv2.line(image, leftIrisMid, leftTop, (0, 255, 0), thickness=2)
  #bottom
  #cv2.line(image, leftIrisMid, leftBottom, (0, 255, 0), thickness=2)
  #cv2.line(image, rightIrisMid, rightBottom, (0, 255, 0), thickness=2)
  #left
  #cv2.line(image, rightIrisLeft, rightLeft, (0, 255, 0), thickness=2)
  #cv2.line(image, leftIrisLeft, leftLeft, (0, 255, 0), thickness=2)
  #right
  #cv2.line(image, leftIrisRight, leftRight, (0, 255, 0), thickness=2)
  #cv2.line(image, rightIrisRight, rightRight, (0, 255, 0), thickness=2)
 

  if distanceCalculator(rightIrisMid,rightBottom) < irisDistcheck and distanceCalculator(leftIrisMid, leftBottom) < irisDistcheck:
    message1 = "looking down"
    #countLag is a variable used to create a bit of a time to make sure that the user has been looking down for a second before the sound is played 
    countLag = countLag + 1
    if countLag >= 3:
        DrowsyDriver = True
        play_sound(DrowsyDriver)  


  elif distanceCalculator(rightIrisLeft, rightLeft) < irisDistcheckLR and distanceCalculator(leftIrisLeft, leftLeft) < irisDistcheckLR:
    message1 = "looking left"
    countLag = 0
  elif distanceCalculator(rightIrisRight, rightRight) < irisDistcheckLR and distanceCalculator(leftIrisRight, leftRight) < irisDistcheckLR:
    message1 = "looking right" 
    countLag = 0
  elif distanceCalculator(rightIrisMid, rightTop) < irisDistcheck and distanceCalculator(leftIrisMid, leftTop) < irisDistcheck:
    message1 = "looking up"
  else:
    message1 = "looking straight"
    countLag = 0
    DrowsyDriver = False
    play_sound(DrowsyDriver)  


  cv2.putText(image, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  print(countLag)
  return countLag

#this function takes the x and y cordinates of 2 points and calculates the distance between them
def distanceCalculator(point1, point2):
  return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5





def play_sound(DrowsyDriver):
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



def steering_check(image):
    GPIO.setmode(GPIO.BCM)
    pin_number = 22
    GPIO.setup(pin_number, GPIO.IN)
    input_value = GPIO.input(pin_number)
    # Print the value
    if not  input_value:
        cv2.putText(image, 'Holding steering :)', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(image, 'Not holding steering :(', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #GPIO.cleanup()


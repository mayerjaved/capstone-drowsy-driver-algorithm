import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
import numpy as np
 


#these variables are used to adjust the eye closed threshold, higher number is more sensitive, lower number is less sensitive
eyeDistcheck = 10
headAnglecheck = 90
irisDistcheck = 8

#the following code is extracted from
#https://www.samproell.io/posts/yarppg/yarppg-face-detection-with-mediapipe/
#the following function takes the captured face mesh landmark coordinates and 
#turns them into a list of tuples representing the (x,y) positions on the image
def get_facemesh_coords(img, landmark_list):
    """Extract FaceMesh landmark coordinates into 468x3 NumPy array.
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
  rightTop = mesh_points[159]
  rightBottom = mesh_points[145]
  distRight = distanceCalculator(rightTop,rightBottom)

  leftTop = mesh_points[386]
  leftBottom = mesh_points[374]
  distLeft = distanceCalculator(leftTop,leftBottom)

  distRatio = (distRight + distLeft)/2

  if (distRatio < eyeDistcheck):
    message = "Eyes: closed"
  else:
    message = "Eyes: open"    
  cv2.putText(image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)









#this function finds the angle between the nose and the 2 eyes and determines the head pitch angle accordingly
def headPitch(image, mesh_points):

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
  else:
    message = "Angle: awake "

  #message += "bottom: " + str(angleC) + " Left:"+ str(angleA) + " Right: " + str(angleB)
  cv2.putText(image, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


#this function will estimate the position of the user's iris position to determine wether they are
#looking stright ahead or no


def eyeIris(image,mesh_points):
  
  rightTopRight = mesh_points[157]
  rightTopRight1 = mesh_points[158]
  rightTopLeft = mesh_points[160]
  rightTopLeft1 = mesh_points[161]

  rightRight = mesh_points[133]
  rightLeft = mesh_points[33]
  rightBottom = mesh_points[145]
  rightTop = mesh_points[159]

  rightBottomLeft = mesh_points[163]
  rightBottomLeft1 =  mesh_points[144]
  rightBottomRight = mesh_points[153]
  rightBottomRight = mesh_points[154]

  rightIrisMid = mesh_points[468]
  rightIrisLeft = mesh_points[471]
  rightIrisRight = mesh_points[469]

  leftRight = mesh_points[263]
  leftLeft = mesh_points[398]
  #leftBottom = mesh_points[]
  leftTop = mesh_points[159]




  if distanceCalculator(rightIrisMid,rightBottom) < irisDistcheck:
    message1 = "looking down"
  elif distanceCalculator(rightIrisLeft, rightLeft) < irisDistcheck:
    message1 = "looking left"
  elif distanceCalculator(rightIrisRight, rightRight) < irisDistcheck:
    message1 = "looking right"
  elif distanceCalculator(rightIrisMid, rightTop) < irisDistcheck:
    message1 = "looking up"
  else:
    message1 = "looking straight"


  cv2.putText(image, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



#this function takes the x and y cordinates of 2 points and calculates the distance between them
def distanceCalculator(point1, point2):
  return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

import cv2
import mediapipe as mp
import math
import numpy as np
import time
import multiprocessing
from playsound import playsound


from facelandmarks import get_facemesh_coords
from facelandmarks import printMesh
from facelandmarks import eyesClosed
from facelandmarks import eyeIris



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

#function to calculate frames per second
def calculate_fps(start_time, frame_count, image):
    end_time = time.time()-start_time
    fps = frame_count/end_time
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(image, fps_text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#main function here:
# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

fps = 0
frame_count = 0
status = False
fps_start_time = time.time()

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

  while cap.isOpened():
    success, image = cap.read()
    #increasing the frame counter by 1 as the camera is running
    frame_count += 1

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    #to resize the frame as the original frame is too big for the screen
    image = cv2.resize(image, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)

    # To improve performance, option1`ally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      #we only want to work with the first face in the image hence
      #results.multi_face_landmarks[0]
      face_landmarks = results.multi_face_landmarks[0]

      #gathers the face mesh points in the form of a tuple array
      mesh_points = get_facemesh_coords(image, face_landmarks)
      eyesClosed(image, mesh_points)
      eyeIris(image, mesh_points)
      calculate_fps(fps_start_time, frame_count, image)
        
      
   

      #eyeIris(image,mesh_points)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh',image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

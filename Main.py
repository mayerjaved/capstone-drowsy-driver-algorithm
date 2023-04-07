import cv2
import mediapipe as mp
import math
import numpy as np
import time
from playsound import playsound
import multiprocessing as mp2

from facelandmarks import getFacemeshCoords
from facelandmarks import printMesh
from facelandmarks import eyesClosed
from facelandmarks import eyeIris
from facelandmarks import faceInCenter

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh




#function to calculate frames per second
def calculateFPS(start_time, frame_count, image):
    end_time = time.time()-start_time
    fps = frame_count/end_time
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(image, fps_text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#this cameraCapture function captures each video frame and puts them in a queue, 
#this function is then later used in multiprocessing where a processor is specifically dedicated to it
def cameraCapture(queue):
    #cap captures frames from the default camera
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        #resizes the camera frames to adjust for different machines
        image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        #the frames go into a queue for processing
        queue.put(image)
    cap.release()





#the imageProcessing function reads the captures frames and processes them. It will capture the landmarks using the mediapipe's library
#We also have other functions added here to print the face landmarks for testing, an FPS calculator to measure performance, eyesClosed function
#to determine if the user has their eyes closed and their head pitch angle, and an eyeIris function to determine which way the user is looking.
def imageProcessing(queue):

    frame_count = 0
    countLagIris = 0
    countLagEyes = 0
    status = False
    fps_start_time = time.time()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while True:
            if not queue.empty():
                frame_count+=1
                image = queue.get()
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    mesh_points = getFacemeshCoords(image, face_landmarks)
                    #printMesh(image, mesh_points)
                    if faceInCenter(image, mesh_points):
                        countLagEyes = eyesClosed(image, mesh_points, countLagEyes)
                        countLagIris = eyeIris(image, mesh_points, countLagIris)
                        calculateFPS(fps_start_time, frame_count, image)
                cv2.imshow('MediaPipe Face Mesh', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break



#This is the main program which creates two processes to run cameraCapture and 
#imageProcessing in seperate processes to implement multiprocessing and improve
#program performance
if __name__ == '__main__':
    queue = mp2.Queue(maxsize=4)
    p1 = mp2.Process(target=cameraCapture, args=(queue,))
    p2 = mp2.Process(target=imageProcessing, args=(queue,))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    #GPIO.cleanup()

def play_sound1():   
    #we use playsound in a in a different tread, as using other methods to play sound results in the function making the program wait 5 sec while the sound finished playing
    # Start playing the sound in a new thread
    sound_thread = threading.Thread(target=playsound, args=("cyber-alarms.mp3",), daemon=True)
    sound_thread.start()

    # Interrupt the execution for 5 seconds
    time.sleep(5)

    # Stop the sound thread
    sound_thread.join()
          


def play_sound2():
    # Initialize pygame
    pygame.init()
    pygame.mixer.init()
    sound = pygame.mixer.Sound("cyber-alarms.mp3")
    sound_thread = threading.Thread(target=sound.play, daemon=True)
    sound_thread.start()
    pygame.time.wait(5000)
    sound.stop()
    sound_thread.join()
    pygame.quit()



def play_sound3(sound_playing):   
   if not sound_playing:
        sound_playing = True
        sound_thread = threading.Thread(target=playsound, args=("cyber-alarms.mp3",), daemon=True)
        sound_thread.start()


#plays sound once and stops
def play_sound():   
    global sound_lock
    
    # try to acquire the threading lock with a timeout of 0.1 seconds
    if sound_lock.acquire(timeout=0.1):
        try:
            # start playing the sound in a new thread
            sound_thread = threading.Thread(target=playsound, args=("arcade.wav",), daemon=True)
            sound_thread.start()

            # wait for the sound thread to finish
            sound_thread.join()
        finally:
            # release the threading lock
            sound_lock.release()


#working
#this function is used to play a warning tone in a different thread
def play_sound1():   
    
#we use playsound in a in a different tread, as using other methods to play sound results in the function making the program wait while the sound finished playing causing a lag in the video
#Start playing the sound in a new thread
    sound_thread = threading.Thread(target=playsound, args=("arcade.wav",), daemon=True)
    sound_thread.start()




#this function will estimate the position of the user's iris position to determine wether they are
#looking stright ahead or no
def eyeIris1(image,mesh_points, countLag):
  
  rightIrisMid = mesh_points[468]
  rightIrisLeft = mesh_points[471]
  rightIrisRight = mesh_points[469]

  rightTopRight = mesh_points[157]
  rightTopRight1 = mesh_points[158]
  rightTopLeft = mesh_points[160]
  rightTopLeft1 = mesh_points[161]

  rightRight = mesh_points[173] #173 innter or 133
  rightLeft = mesh_points[130] #33 inner or 130 outer
  rightBottom = mesh_points[145]
  rightTop = mesh_points[159]

  rightBottomLeft = mesh_points[163]
  rightBottomLeft1 =  mesh_points[144]
  rightBottomRight = mesh_points[153]
  rightBottomRight = mesh_points[154]

  leftIrisMid = mesh_points[473]
  leftIrisLeft = mesh_points[476]
  leftIrisRight = mesh_points[474]

  leftRight = mesh_points[263] #466 or 263
  leftLeft = mesh_points[414] #398 or 414 04 413
  leftBottom = mesh_points[374]
  leftTop = mesh_points[386]


  #we have a few line drawings for testing to make sure the right points from the 
  #top
  cv2.line(image, rightIrisMid, rightTop, (0, 255, 0), thickness=2)
  cv2.line(image, leftIrisMid, leftTop, (0, 255, 0), thickness=2)
  #bottom
  cv2.line(image, leftIrisMid, leftBottom, (0, 255, 0), thickness=2)
  cv2.line(image, rightIrisMid, rightBottom, (0, 255, 0), thickness=2)
  #left
  cv2.line(image, rightIrisLeft, rightLeft, (0, 255, 0), thickness=2)
  cv2.line(image, leftIrisLeft, leftLeft, (0, 255, 0), thickness=2)
  #right
  cv2.line(image, leftIrisRight, leftRight, (0, 255, 0), thickness=2)
  cv2.line(image, rightIrisRight, rightRight, (0, 255, 0), thickness=2)
 

  if distanceCalculator(rightIrisMid,rightBottom) < irisDistcheck and distanceCalculator(leftIrisMid, leftBottom) < irisDistcheck:
    message1 = "looking down"
    #countLag is a variable used to create a bit of a time to make sure that the user has been looking down for a second before the sound is played 
    countLag = countLag + 1
    if countLag >= 7:
        DrowsyDriver = True
        playSound()  
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


  cv2.putText(image, message1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  print(countLag)
  return countLag



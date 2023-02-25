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
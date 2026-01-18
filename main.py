import cv2
from screeninfo import get_monitors

# different brainrots
blank = cv2.imread('imgs/blank.gif')
frown = cv2.imread('imgs/frown.jpg')
smile_img = cv2.imread('imgs/smile.jpg')
smirk = cv2.imread('imgs/smirk.jpg')

# detectors for faces n stuff
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# using opencv boilerplate code
camera = cv2.VideoCapture(0)

# find screen dimensions
my_monitor = get_monitors()[0]
half_screen_width = my_monitor.width // 2

while(True): 
    # Capture frame-by-frame
    ret, my_frame = camera.read()
    my_frame = cv2.flip(my_frame, 1)
    
    # make half screen without stretching
    vid_height, vid_width = my_frame.shape[:2]
    aspect_ratio = vid_height / vid_width
    half_screen_height = int(half_screen_width * aspect_ratio)
    camera_feed = cv2.resize(my_frame, (half_screen_width, half_screen_height))
    
    # face and smile detection
    grayscale = cv2.cvtColor(my_frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector.detectMultiScale(grayscale, 1.3, 5)
    
    showing = blank
    for (x, y, w, h) in detected_faces:
        face_area = grayscale[y:y+h, x:x+w]
        
        # check for full smile
        found_smiles = smile_detector.detectMultiScale(face_area, 1.8, 20)
        if len(found_smiles) > 0:
            showing = smile_img
            break
        
        # smirk? or nah
        mouth_area = face_area[int(h*0.5):, :]  # Bottom half of face
        left_side = mouth_area[:, :mouth_area.shape[1]//2]
        right_side = mouth_area[:, mouth_area.shape[1]//2:]
        
        left_smile = smile_detector.detectMultiScale(left_side, 1.3, 3)
        right_smile = smile_detector.detectMultiScale(right_side, 1.3, 3)
        
        # smirk detected if smile on one side but not the other
        if (len(left_smile) > 0 and len(right_smile) == 0) or (len(left_smile) == 0 and len(right_smile) > 0):
            showing = smirk
            break
        
        # check for squinting eyes
        found_eyes = eye_detector.detectMultiScale(face_area, 1.05, 3)
        if len(found_eyes) >= 2:
            # calculate average eye height
            total_eye_height = sum([ey_h for (ey_x, ey_y, ey_w, ey_h) in found_eyes]) / len(found_eyes)
            squint_level = total_eye_height / h
            
            if squint_level < 0.10:  
                showing = frown
                break
    
    brainrot_to_show = cv2.resize(showing, (half_screen_width, half_screen_height))
    cv2.imshow("Displayed Image", brainrot_to_show)

    # Display the resulting frame
    cv2.imshow('frame', camera_feed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
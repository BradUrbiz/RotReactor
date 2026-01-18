import cv2
from screeninfo import get_monitors

# different brainrots
blank = cv2.imread('imgs/blank.gif')
frown = cv2.imread('imgs/frown.jpg')
smile = cv2.imread('imgs/smile.jpg')
smirk = cv2.imread('imgs/smirk.jpg')

# using opencv boilerplate code
cap = cv2.VideoCapture(0)

# find screen dimensions
monitor = get_monitors()[0]
target_width = monitor.width // 2

while(True): 
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # make half screen without stretching
    height, width = frame.shape[:2]
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    frame = cv2.resize(frame, (target_width, target_height))
    
    blank = cv2.resize(blank, (target_width, target_height))
    cv2.imshow("Displayed Image", blank)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
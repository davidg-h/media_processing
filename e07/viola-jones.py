import os
import cv2
from cv2 import VideoCapture

#python program to check if a path exists
#if it doesn’t exist we create one
if not os.path.exists('frames'):
   os.makedirs('frames')

cam = VideoCapture(0)

haarcascade_root = r'data/haarcascades'

counterx = 0
countery = 0

while True:
    # get a frame
    ret, frame = cam.read()

    # show frame
    # cv2.imshow("Webcam-Video", frame)

    # grey scale covertion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # saving for cropping
    croppedImg = frame

    # face recog
    face_cascade = cv2.CascadeClassifier(haarcascade_root + r"/haarcascade_frontalface_default.xml")
    detect_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, width, height) in detect_faces:
        centerCoords = (int(x+width/2), int(y+height/2.5))
        axesLength = (int(width/2), int(height/1.3))
        
        croppedImg = croppedImg[y - (int(height/1.3/2)) :y - (int(height/1.3/2)) +(int(height/1.3))*2, x - (int(height/1.3/2)): x - (int(height/1.3/2))+(int(height/1.3))*2]
        cv2.ellipse(
            frame,
            centerCoords,
            axesLength,
            0,
            0,
            360,
            (0, 255, 0),
            2
        )
    
    cv2.imshow("Webcam-Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s') and countery < 10:
        if ret:
            # img_resize = cv2.resize(frame, (0,0), fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
            filename = f"frames\\face" + ".jpg"

            scale_percent = 66
            width = int(croppedImg.shape[1]*scale_percent/100)
            height = int(croppedImg.shape[0]*scale_percent/100)
            resized = cv2.resize(croppedImg, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(filename, resized)

            if counterx % 10 == 0:
                countery+=1
                
    # quit with q
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
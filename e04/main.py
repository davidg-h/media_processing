# import the opencv library
import cv2  

def on_trackbar(value):
    print(value)

def showCaps(filename):
    img = cv2.imread(filename)
    cv2.imshow("cap_frame", img)
    # add trackbar for contrast and lighting
    initial_value = 50
    max_value = 100
    cv2.createTrackbar('contrast', 'cap_frame', initial_value, max_value, on_trackbar)
    
  
# define a video capture object
vid = cv2.VideoCapture(0)

# frame counter
counter = 0
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
    # half resoluzion of frame
    frame_resized = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC)
    # Display the resulting frame
    cv2.imshow('video_feed', frame)
    
    
    
    key = cv2.waitKey(1) & 0xFF # get the lowest 8 bits of the keycode
    
    if key == ord("q"):
         # the 'q' button is set as the
        # quitting button you may use any
        break
    elif key == ord("s"):
        filename = f"frames/Frame_{counter}.jpg"
        if ret and counter < 100:
            print("Image saved")
            cv2.imwrite(filename,frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
            # show saved frames in seperate windows
            showCaps(filename)
            counter += 1
   
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
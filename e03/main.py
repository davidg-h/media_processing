import cv2
import numpy as np

# read in an image and display it in a window
image = cv2.imread("rose.jpg")

# get the rgb value of a pixel on click 
def mouseRGB(event,x,y,flags,param):
    global image
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        # Clear previous text from the image
        image = cv2.imread("rose.jpg")
        
        colorsB = image[y,x,0]
        colorsG = image[y,x,1]
        colorsR = image[y,x,2]
        print("Red: ",colorsR)
        print("Green: ",colorsG)
        print("Blue: ",colorsB)
        print("Coordinates of pixel: X: ",x,"Y: ",y)
        # Convert RGB to BGR
        bgr = (colorsB, colorsG, colorsR)
        # Convert BGR to HSV
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        print("Hue:", hsv[0])
        print("Saturation:", hsv[1])
        print("Value:", hsv[2])
       # Display the RGB value as text on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"RGB:{colorsR},{colorsG},{colorsB}\nHSV:{hsv[0]},{hsv[1]},{hsv[2]}"
        font_scale = 0.5
        font_thickness = 1
        
        # Get the size of the text
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Calculate the maximum font size that can fit within the image
        max_font_scale = min(image.shape[0] / text_height, image.shape[1] / text_width)
        font_scale *= max_font_scale
        
        # Recalculate the size of the text with the new font size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Calculate the position of the text so that it's centered on the clicked pixel
        x_pos = max(0, x - text_width // 2)
        y_pos = max(text_height, y - 10)
        
        # Display the text on the image
        cv2.putText(image, text, (x_pos, y_pos), font, font_scale, (255, 255, 255), font_thickness)
   
# Read an image, a window and bind the function to window 
cv2.namedWindow('mouseRGB')
cv2.setMouseCallback('mouseRGB',mouseRGB)

# Do until esc pressed
while(1):
    cv2.imshow('mouseRGB',image)
    if cv2.waitKey(20) & 0xFF == 27:
        break
        
# If esc pressed, finish.
cv2.destroyAllWindows()

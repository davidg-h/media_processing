# import the opencv library
import cv2
import numpy as np
from Operators import Operator


def change_contrast(val):
    global contrast
    contrast = val
    perform_operation(img)


def change_brightness(val):
    global brightness
    brightness = val / 250
    perform_operation(img)


def perform_operation(img):
    im1 = img * contrast + brightness
    cv2.imshow("cap_frame", im1)


def showCaps(filename):
    global trackbar_created
    global img
    img = cv2.imread(filename)
    img = np.float32(img / 255)
    cv2.imshow("cap_frame", img)

    if not trackbar_created:
        # add trackbar for contrast and brightness
        contrast = 1
        brightness = 0
        cv2.createTrackbar("contrast", "cap_frame", contrast, 3, change_contrast)
        cv2.createTrackbar(
            "brightness", "cap_frame", brightness, 200, change_brightness
        )
        trackbar_created = True


def imgOperations(filename):
    img = cv2.imread(filename)
    # converting to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # remove noise
    img = cv2.GaussianBlur(gray, (3, 3), 0)
    Operator.Laplacian(img)
    Operator.Sobel(img)
    Operator.Canny(img)


trackbar_created = False

# define a video capture object
vid = cv2.VideoCapture(0)

# frame counter
counter = 0

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # half resoluzion of frame
    frame_resized = cv2.resize(
        frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC
    )
    # Display the resulting frame
    cv2.imshow("video_feed", frame)

    key = cv2.waitKey(1) & 0xFF  # get the lowest 8 bits of the keycode

    if key == ord("q"):
        # the 'q' button is set as the
        # quitting button you may use any
        break
    elif key == ord("s"):
        filename = f"frames/Frame_{counter}.jpg"
        if ret and counter < 100:
            print("Image saved")
            cv2.imwrite(filename, frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
            # show saved frames in seperate windows
            showCaps(filename)
            imgOperations(filename)
            counter += 1

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

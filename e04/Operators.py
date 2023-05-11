import cv2


class Operator:
    @staticmethod
    def Laplacian(img):
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        cv2.imshow("Laplacian", laplacian)

    @staticmethod
    def Sobel(img):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        cv2.imshow("Sobel-X", sobelx)
        cv2.imshow("Sobel-Y", sobely)

    @staticmethod
    def Canny(img):
        # Setting parameter values
        t_lower = 50  # Lower Threshold
        t_upper = 150  # Upper threshold
        # Applying the Canny Edge filter
        edge = cv2.Canny(img, t_lower, t_upper)
        cv2.imshow("Canny", edge)

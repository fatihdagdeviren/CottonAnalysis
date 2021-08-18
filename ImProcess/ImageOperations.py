import cv2
from Utilities import Cons
import numpy as np
#region MaskBounds

whiteSensivity = 50
lowerWhiteForMask = (0, 0, 255 - whiteSensivity)
upperWhiteForMask = (255, 255 - whiteSensivity, 255)
# find the green color
lowerGreenForMask = (36, 0, 0)
upperGreenForMask = (86, 255, 255)
# find the brown color
lowerBrownForMask = (8, 60, 20)
upperBrownForMask = (30, 255, 200)
# find the yellow color
lowerYellowForMask = (21, 39, 64)
upperYellowForMask = (40, 255, 255)

# morphological Operations
kernel = np.ones((5, 5),np.uint8)
iterations = 1
#endregion


"""
    Apply Color mask on image usişng hsv color space (default: whiteMask)
"""
def AppyMaskToImage(image, lowerBounds = [lowerWhiteForMask], upperBounds = [upperWhiteForMask]):
    try:
        imageNew = image.copy()
        hsvImage = cv2.cvtColor(imageNew, cv2.COLOR_BGR2HSV)
        bounds = list(zip(lowerBounds, upperBounds))
        mask = cv2.inRange(hsvImage, bounds[0][0], bounds[0][1])
        if len(bounds) > 1:
            for bound in bounds[1:]:
                # Bitwise-AND mask and original image
                maskNew = cv2.inRange(hsvImage, bound[0], bound[1])
                mask = cv2.bitwise_or(mask, maskNew)
        res = cv2.bitwise_and(imageNew, imageNew, mask=mask)
        return Cons.SucessVal, [res, mask]
    except BaseException as e:
        return Cons.ErrorVal, "{0}-{1}".format(Cons.ErrorMessage, str(e))

def Erode(image):
    try:
        imageNew = image.copy()
        erosion = cv2.erode(imageNew, kernel, iterations= iterations)
        return Cons.SucessVal, erosion
    except BaseException as e:
        return Cons.ErrorVal, "{0}-{1}".format(Cons.ErrorMessage, str(e))

def Dilate(image):
    try:
        imageNew = image.copy()
        dilation = cv2.dilate(imageNew, kernel, iterations= iterations)
        return Cons.SucessVal, dilation
    except BaseException as e:
        return Cons.ErrorVal, "{0}-{1}".format(Cons.ErrorMessage, str(e))

"""
    Dilation followed by Erosion.
"""
def Opening(image):
    try:
        imageNew = image.copy()
        opening = cv2.morphologyEx(imageNew, cv2.MORPH_OPEN, kernel)
        return Cons.SucessVal, opening
    except BaseException as e:
        return Cons.ErrorVal, "{0}-{1}".format(Cons.ErrorMessage, str(e))

"""
    erosion followed by dilation
"""
def Closing(image):
    try:
        imageNew = image.copy()
        closing = cv2.morphologyEx(imageNew, cv2.MORPH_CLOSE, kernel)
        return Cons.SucessVal, closing
    except BaseException as e:
        return Cons.ErrorVal, "{0}-{1}".format(Cons.ErrorMessage, str(e))

"""
    RGB goruntudeki tum katmanlar renk bileşeni içerdiği için ilk olarak COLOR_BGR2YCrCb uzayına döndürülür resim. 
    Y alanı üzerinde Equalization yapıldıktan sonra tekrar RGB ye dondurulmektedir.
"""
def ApplyHistogramEqualizationToColoredImage(image):
    try:
        imgNew = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        imgNew[:, :, 0] = cv2.equalizeHist(imgNew[:, :, 0])
        resImage = cv2.cvtColor(imgNew, cv2.COLOR_YCrCb2BGR)
        return Cons.SucessVal, resImage
    except BaseException as e:
        return Cons.ErrorVal, "{0}-{1}".format(Cons.ErrorMessage, str(e))
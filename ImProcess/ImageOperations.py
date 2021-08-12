import cv2
from Utilities import Cons

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
        return [res, mask], Cons.SucessMessage
    except BaseException as e:
        return Cons.ErrorVal, "{0}-{1}".format(Cons.ErrorMessage, str(e))



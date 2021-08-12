
import cv2
import math
import numpy as np

class ContourFinder():

    def __init__(self, cannyMin, cannyMax, myImage, waitKey, imshow,convertToGray=False):
        self.cannyMin = cannyMin
        self.cannyMax = cannyMax
        self.contourList = []
        self.boxes = []
        self.waitKey = waitKey
        self.image = myImage
        self.imshow = imshow
        self.convertToGrayScale = convertToGray

    # @property
    # def image(self):
    #     return self.image()
    #
    # @image.setter
    # def image(self, newImage):
    #     self.image = newImage
    #     if not self.CheckIfImageIsGrayScale():
    #         self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    #
    # @image.deleter
    # def image(self):
    #     del self.image

    """
        Goruntu uzerindeki contourlari bulmaya yarayan fonksiyon.
    """
    def GetContoursFromImage(self):
        try:
            self.contourList = []
            if not self.CheckIfImageIsGrayScale() and self.convertToGrayScale:
               self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            roiCany = cv2.Canny(self.image, self.cannyMin, self.cannyMax)
            (roiCnts, roiNew) = cv2.findContours(roiCany, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            roiCntsAreas = []
            for roiCnt in roiCnts:
                x, y, w, h = cv2.boundingRect(roiCnt)
                roiCntArea = self.image[y:y + h, x:x + w]
                roiCntsAreas.append([x, y, w, h, roiCntArea, roiCnt])
            self.contourList = roiCntsAreas
        except BaseException as e:
            return None
    """
        Verilen iki nokta arasında aci bulmaya yaran fonksiyon.
    """
    def GetAngle(self, p1, p2):
        return math.atan2(p1[1] - p2[1], p1[0] - p2[0]) * 180 / math.pi

    """
        Goruntudeki contourlari box yapisina ceviren fonksiyon.
    """
    def DrawBoxes(self):
        self.boxes = []
        for contour in self.contourList:
            minRect = cv2.minAreaRect(contour[5])
            box = cv2.boxPoints(minRect)
            box = np.int0(box)
            self.boxes.append(box)
    """
        Contourlari Goruntu üzerinde cizimini yapan fonksiyon.
    """
    def DrawContours(self,winName):
        tempImage = self.image.copy()
        if len(self.boxes) > 0:
            for box in self.boxes:
                cv2.drawContours(tempImage, [box], 0, (0, 255, 0), 1)
        if self.imshow:
            cv2.imshow(winName, tempImage)
            if isinstance(self.waitKey, int):
                cv2.waitKey(self.waitKey)
    """ 
        Goruntunun kanal sayisini kontrol eden fonksiyon        
    """
    # check If image is single channel
    def CheckIfImageIsGrayScale(self):
        return len(self.image) == 1

    """
        Contourlari NMS algoritmasından geciren fonksiyon.
        Ref: https: // www.pyimagesearch.com / 2014 / 11 / 17 / non - maximum - suppression - object - detection - python /
        https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        0.3 <= overlapThresh <= 0.5
    """
    def CalculateNonMaximumSupression(self, overlapThresh):
        # Malisiewicz et al.
        # initialize the list of picked indexes
        boxes = np.array(self.contourList)
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = np.array(boxes[:, 0], dtype=float)
        y1 = np.array(boxes[:, 1], dtype=float)
        x2 = np.array(boxes[:, 0] + boxes[:, 2], dtype=float)
        y2 = np.array(boxes[:, 1] + boxes[:, 3], dtype=float)
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked
        # return boxes[pick] # boxlari doner
        self.contourList = [self.contourList[x] for x in pick]
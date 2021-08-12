import cv2
from DLModel import Model as myModel
from ImProcess import Contour as contourFinder
from FileOperations import FileOperations
from DLModel import Yolo as yolo
import numpy as np
from ImProcess import ImageOperations as imop
if __name__ == "__main__":

    #region ModelOlusturmaOrnek

    # myModel = myModel.MyDLM()
    # model = myModel.GetModelFromFile()
    # if model is not None:
    #     image = cv2.imread('PetImages/Cat/102.jpg')
    #     resized = cv2.resize(image, (32,32), interpolation = cv2.INTER_AREA)
    #     resized = resized.reshape(1, 32, 32, 3).astype('float32')
    #     resized /= 255
    #     predictions = model.predict(resized)
    #     print(predictions)
    #     x=2

    #endregion

    #region ContourKullanimiOrnek

    # image = cv2.imread('Temp\Images\cotton1.jpeg')
    #
    # contFinder = contourFinder.ContourFinder(100, 200, image, waitKey=0, imshow=True, convertToGray=False)
    #
    # contFinder.GetContoursFromImage()
    # contFinder.DrawBoxes()
    # contFinder.DrawContours("Contours")
    # NmsPoints = contFinder.CalculateNonMaximumSupression(0.3)
    # contFinder.DrawBoxes()
    # contFinder.DrawContours("NMS-03")
    #
    # contFinder.GetContoursFromImage()
    # NmsPoints = contFinder.CalculateNonMaximumSupression(0.5)
    # contFinder.DrawBoxes()
    # contFinder.DrawContours("NMS-05")

    #endregion


    #region FileOperationsTest

    # image = cv2.imread('cotton.jpg')
    # res, message = FileOperations.CreatePickleFromFile("Merhaba.pkl", image)
    # res2, object = FileOperations.LoadPickleFromFile("Merhaba.pkl")

    #endregion

    #region YOLO

    # yoloModel = yolo.Yolo('Temp\Images\cotton1.jpeg')
    # # yoloModel.PrintLayerNames()
    # yoloModel.Process()
    # yoloModel.DrawIndices()
    # cv2.imshow("ProcessedImage", yoloModel.processedImage)
    # cv2.waitKey(0)

    #endregion

    #region MaskCotton

    image = cv2.imread('Temp\Images\cotton3.jpg')
    # retImage, _ = imop.AppyMaskToImage(image,
    #                                    [imop.lowerBrownForMask, imop.lowerGreenForMask, imop.lowerYellowForMask],
    #                                    [imop.upperBrownForMask, imop.upperGreenForMask, imop.upperYellowForMask])

    retImage, _ = imop.AppyMaskToImage(image,
                                       [imop.lowerWhiteForMask],
                                       [imop.upperWhiteForMask])

    contFinder = contourFinder.ContourFinder(100, 200, retImage[0], waitKey=1, imshow=True, convertToGray=False)
    contFinder.GetContoursFromImage()
    contFinder.DrawBoxes()
    contFinder.DrawContours("Contours")
    contFinder.CalculateNonMaximumSupression(0.3)
    contFinder.DrawBoxes()
    contFinder.DrawContours("NMS-03")

    contFinder.GetContoursFromImage()
    NmsPoints = contFinder.CalculateNonMaximumSupression(0.5)
    contFinder.DrawBoxes()
    contFinder.DrawContours("NMS-05")


    cv2.imshow('Org. Resim', image)
    cv2.imshow('retImage', retImage[0])
    cv2.imshow('retImageMask', retImage[1])
    cv2.waitKey(0)

    #endregion

    print("Son")


# -*- coding: utf-8 -*-
'''
Deep Learning Türkiye topluluğu tarafından hazırlanmıştır.
Amaç: Fotoğraftaki nesneyi sınıflandırmak.
Veriseti: CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html)
Algoritma: Evrişimli Sinir Ağları (Convolutional Neural Networks)
Eğer arkaplanda TensorFlow kullanıyorsanız otomatik olarak GPU kullanılacaktır.
Theano arkaplanında GPU kullanarak çalıştırmak için gereken komut:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py
25 epoch sonunda hata oranı 0.65'e, 50 epoch sonunda 0.55'e düşürüldü.
(Bu durumda hala yetersiz durumda.)
'''


# Label 	Description
# 0 	airplane
# 1 	automobile
# 2 	bird
# 3 	cat
# 4 	deer
# 5 	dog
# 6 	frog
# 7 	horse
# 8 	ship
# 9 	truck



import cv2
from DLModel import Model as myModel
from ImProcess import Contour as contourFinder
from FileOperations import FileOperations


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

    # image = cv2.imread('cotton.jpg')
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

    image = cv2.imread('cotton.jpg')
    res, message = FileOperations.CreatePickleFromFile("Merhaba.pkl", image)
    res2, object = FileOperations.LoadPickleFromFile("Merhaba.pkl")

    #endregion


    print("Son")


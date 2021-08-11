import cv2
import numpy as np

yoloCfgPath = 'D:\OutSource\CottonAnalysis\DLModel\WeightsAndConfs\yolov3.cfg'
yoloWeightsPath = 'D:\OutSource\CottonAnalysis\DLModel\WeightsAndConfs\yolov3.weights'
cocoNamesPath = 'D:\OutSource\CottonAnalysis\DLModel\WeightsAndConfs\coco.names'
#https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html

class Yolo():
    def __init__(self, imagePath):
        # Load names of classes and get random colors
        self.image = cv2.imread(imagePath)
        self.classes = open(cocoNamesPath).read().strip().split('\n')
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')
        # Give the configuration and weight files for the model and load the network.
        self.net = cv2.dnn.readNetFromDarknet(yoloCfgPath, yoloWeightsPath)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        # determine the output layer
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.indices = None
        self.conf = 0.5
        self.boxes = []
        self.confidences = []
        self.classIDs = []
        self.processedImage = None

    def PrintLayerNames(self):
        ln = self.net.getLayerNames()
        print(len(ln), ln)

    def Process(self):
        self.boxes = []
        self.confidences = []
        self.classIDs = []
        blob = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)

        # combine the 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)

        H, W = self.image.shape[:2]

        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > self.conf:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w // 2), int(y - h // 2)
                p1 = int(x + w // 2), int(y + h // 2)
                self.boxes.append([*p0, int(w), int(h)])
                self.confidences.append(float(confidence))
                self.classIDs.append(classID)
                # cv.rectangle(img, p0, p1, WHITE, 1)
        self.indices = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.conf, self.conf - 0.1)


    def DrawIndices(self):
        self.processedImage = self.image.copy()
        if len(self.indices) > 0 and self.processedImage is not None:
            for i in self.indices.flatten():
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])
                color = [int(c) for c in self.colors[self.classIDs[i]]]
                cv2.rectangle(self.processedImage, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.classes[self.classIDs[i]], self.confidences[i])
                cv2.putText(self.processedImage, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
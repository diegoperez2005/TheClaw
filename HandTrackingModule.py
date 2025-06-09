import cv2
import mediapipe as mp
import time

#Framework being used is mediapipe by google, helps you get quickly started with ML Solutions:
#like hand tracking, object detection, box tracking etc.
#
#I'll be using hand-tracking module
#Broken down into two components: Palm Detection and Hand landmarks
#Palm detection focuses on getting a general image and cropping the image of the hand
#hand-landmarks finds 21 different points on the hand and marks where each of them are

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True ):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
         # this is creating the hand image displaying

        if results.multi_hand_landmarks:
             for handLms in results.multi_hand_landmarks:
                 if draw:
                     self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
                #What this does is that it draws out the connection between each of the hand landmarksg
                #for id, lm in enumerate(handLms.landmark):
                    # this gets the index numbers and coordinates of each of the hand landmark points
                    # print(id,lm)
                    #h, w, c = img.shape
                    # height, width and channel of landmark
                    #cx, cy = int(lm.x * w), int(lm.y * h)
                    #print(id, ':', cx, cy)
                    # if id == 4:
                    #cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 15), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()


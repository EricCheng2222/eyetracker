# import the opencv library
import cv2
import numpy as np
from gaze_tracking import GazeTracking
  

def getLeftMostEye(eyes):
    leftmost=9999999
    leftmostindex=-1
    size = len(eyes)
    for i in range(0,size):
        if eyes[i][0]<leftmost:
            leftmost=eyes[i][0]
            leftmostindex=i
    return [eyes[leftmostindex]]
  
# define a video capture object
gaze = GazeTracking()
vid = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')  
MaxR = 20
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = face_cascade.detectMultiScale(gray, 1.1, 3)
    leftMost = getLeftMostEye(eyes)

    for (x, y, w, h) in leftMost:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cut = frame[y:y+h, x:x+w] 
        replace = frame[y:y+h, x:x+w]
        frame[0:h, 0:w, :] = replace

    grayEye = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
    (_, blackAndWhiteImage) = cv2.threshold(grayEye, 75, 255, cv2.THRESH_BINARY)
    grayEyeBlur = cv2.medianBlur(grayEye, 9)
    row = grayEyeBlur.shape[0]
    circles = cv2.HoughCircles(grayEyeBlur, cv2.HOUGH_GRADIENT, 1.0, row/16, param1=100, param2=18, minRadius=5, maxRadius=MaxR)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(circles)
        for (x, y ,r) in circles:
            cv2.circle(cut, (x, y), r, (0, 255, 0), 1)
            blackAndWhiteImage = blackAndWhiteImage[y-r:y+r, x-r:x+r]
    # Display the resulting frame

    cv2.imshow('frame', frame)
    cv2.imshow('eye', cut) 
    if blackAndWhiteImage.shape[0]>0 and blackAndWhiteImage.shape[1]>0:
        cv2.imshow('bw', blackAndWhiteImage)
    cv2.waitKey(10)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

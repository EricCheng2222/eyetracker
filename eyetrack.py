# import the opencv library
import cv2
import numpy as np
from gaze_tracking import GazeTracking
  
#FIXME might go out of range since eye is possible not present
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
MaxR = 60
cnt = 0
GREEN = (0, 255, 0)
ReflecX, ReflecY = 0, 0
avg_left , sum_left  = 0, 0
avg_mid  , sum_mid   = 0, 0
avg_right,sum_right  = 0, 0

while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    gaze.refresh(frame)
    anno_frame = gaze.annotated_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #eyes = face_cascade.detectMultiScale(gray, 1.1, 3)
    #leftMost = getLeftMostEye(eyes)
    left_pupil = gaze.pupil_left_coords()
    cut = frame[left_pupil[1]-MaxR:left_pupil[1]+MaxR, left_pupil[0]-MaxR:left_pupil[0]+MaxR]
    frame[0:2*MaxR, 0:2*MaxR, :] = cut
    #for (x, y, w, h) in leftMost:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #    cut = frame[y:y+h, x:x+w] 
    #    frame[0:h, 0:w, :] = cut

    grayEye = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
    (_, blackAndWhiteImage) = cv2.threshold(grayEye, 80, 255, cv2.THRESH_BINARY)
    #grayEyeBlur = cv2.medianBlur(grayEye, 9)
    #row = grayEyeBlur.shape[0]
    #circles = cv2.HoughCircles(grayEyeBlur, cv2.HOUGH_GRADIENT, 1.0, row/16, param1=100, param2=18, minRadius=int(0.8*MaxR), maxRadius=MaxR)

    #if circles is not None:
    #    circles = np.round(circles[0, :]).astype("int")
    #    #print(circles)
    #    for (x, y ,r) in circles:
    #        cv2.circle(cut, (x, y), r, GREEN, 1)
    #        blackAndWhiteImage = blackAndWhiteImage[y-r:y+r, x-r:x+r]
    
    BW_width = blackAndWhiteImage.shape[0]
    if BW_width > 0 and BW_width <= 2*MaxR:
        BW_circles = cv2.HoughCircles(blackAndWhiteImage, cv2.HOUGH_GRADIENT, 1.0, 4, param1=100, param2=6, minRadius=2, maxRadius=int(MaxR*0.15))
        blackAndWhiteImage = cv2.cvtColor(blackAndWhiteImage,cv2.COLOR_GRAY2RGB)
        if BW_circles is not None:
            BW_circles = np.round(BW_circles[0, :]).astype("int")
            #print(BW_circles)
            for (x, y, r) in BW_circles:
                ReflecX, ReflecY = x, y
                #print("X:" + str(int(BW_width/2)-ReflecX))
                #print("Y:" + str(int(BW_width/2)-ReflecY))
                cv2.circle(blackAndWhiteImage, (x, y), r, GREEN, 1)

    #callibration
    #if cnt < 30:
    #    #print("MID")
    #    cv2.circle(frame, (960, 540), 20, GREEN, 5)
    #    cnt = cnt+1
    #    sum_mid += (int(BW_width/2)-ReflecX)
    #elif cnt < 60:
    #    #print("LEFT")
    #    cv2.circle(frame, (0, 540), 20, GREEN, 5)
    #    cnt = cnt+1
    #    sum_left += (int(BW_width/2) - ReflecX)
    #elif cnt < 90:
    #    #print("RIGHT")
    #    cv2.circle(frame, (1920, 540), 20, GREEN, 5)
    #    cnt = cnt+1
    #    sum_right += (int(BW_width/2) - ReflecX)
    #else:    
    #    #redundant calculation, can be eliminate
    #    avg_left  = int(sum_left/30)     
    #    avg_right = int(sum_right/30)
    #    #print("AVGL:" + str(avg_left))
    #    #print("AVGR:" + str(avg_right))
    #    scale = int(1920/(avg_left-avg_right))
    #    print("SCALE:" + str(scale))
    #    cv2.circle(frame, (scale*(avg_left-(int(BW_width)-ReflecX)), 540), 20, GREEN, 5)
    #    if abs(scale)>75:
    #        cnt = 0

    #print(cnt)
    # Display the resulting frame
    cv2.imshow('frame', anno_frame)
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

# import the opencv library
import cv2
import numpy as np
  
  
# define a video capture object
vid = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')  
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = face_cascade.detectMultiScale(gray, 1.1, 3)
    flag = False
    cut = 0
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if flag is False:
            flag = True
            cut = frame[y:y+h, x:x+w] 
        replace = frame[y:y+h, x:x+w]
        frame[0:h, 0:w, :] = replace
    grayEye = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
    grayEye = cv2.medianBlur(grayEye, 7)
    #(thresh, blackAndWhiteImage) = cv2.threshold(grayEye, 100, 255, cv2.THRESH_BINARY)
    row = grayEye.shape[0]
    circles = cv2.HoughCircles(grayEye, cv2.HOUGH_GRADIENT, 1.0, row/16, param1=100, param2=18, minRadius=1, maxRadius=21)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(circles)
        for (x, y ,r) in circles:
            cv2.circle(cut, (x, y), r, (0, 255, 0), 3)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('eye', cut) 
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

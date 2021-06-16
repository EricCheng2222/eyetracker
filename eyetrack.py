# import the opencv library
import cv2
  
  
# define a video capture object
vid = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')  
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    flag = False
    cut = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if flag is False:
            flag = True
            cut = frame[y:y+h, x:x+w] 
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

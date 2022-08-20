import cv2

# Use the xml files from haarcascade to help in detecting the faces and eyes
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Open the device's webcame to capture vide
captureWebcam = cv2.VideoCapture(0)

while True:
    # take the video's frame
    _, videoFrame = captureWebcam.read()
    # to detect the faces and eyes, the image must be transformed into grayscale
    transformToGrayscale = cv2.cvtColor( videoFrame, cv2.COLOR_BGR2GRAY )

    # detect the faces and eyes using the scalefactor 1.1/1.2 and minNeighbors 4
    detectedFaces = faceCascade.detectMultiScale( transformToGrayscale, 1.1, 4)
    detectedEyes = eyesCascade.detectMultiScale(transformToGrayscale, 1.1, 4)

    # use the coordinates to draw a rectangle on the detected faces and eyes
    countFaces = len( detectedFaces) 
    countEyes = len( detectedEyes) 
    loopIteration = 0
    while True:
        if loopIteration < countFaces:
            (x, y, w, h) = detectedFaces[loopIteration]
            cv2.rectangle( videoFrame, (x, y), (x+w, y+h), ( 0, 255, 0 ), 2)
        if loopIteration < countEyes:
            (x, y, w, h) = detectedEyes[loopIteration]
            cv2.rectangle( videoFrame,(x,y),(x+w,y+h),(255, 0, 0),2)
        loopIteration += 1
        if loopIteration > countFaces and loopIteration > countEyes:
            break

    # An alternate solution?
    # But this uses two loops so it might be a lot slower
    #for (x, y, w, h) in detectedFaces:
    #    cv2.rectangle( videoFrame, (x, y), (x+w, y+h), ( 0, 255, 0 ), 2)
    #for (x,y,w,h) in detectedEyes:
    #    cv2.rectangle( videoFrame,(x,y),(x+w,y+h),(255, 0, 0),2)
    # Output the result
    cv2.imshow( "Face and Eyes Recognition", videoFrame )

    # To end the program, press escape
    exitESC = cv2.waitKey(30) & 0xff
    if exitESC == 27:
        break

# Terminate webcam capture
captureWebcam.release()
cv2.destroyAllWindows()

# Additional comments:
# Finished on 3:30pm
# I'll try to deploy this on a website

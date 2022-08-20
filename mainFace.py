import cv2

# Open the device's webcame to capture vide
captureWebcam = cv2.VideoCapture(0)
# Use the xml files from haarcascade to help in detecting the faces and eyes
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

class webcamVideo():
    def __init__(self):
        pass

    def __del__(self):
        # Terminate webcam capture
        captureWebcam.release()

    def getFrame(self):

        # take the video's frame
        _, self.videoFrame = captureWebcam.read()
        # to detect the faces and eyes, the image must be transformed into grayscale
        self.transformToGrayscale = cv2.cvtColor( self.videoFrame, cv2.COLOR_BGR2GRAY )

        # detect the faces and eyes using the scalefactor 1.1 and minNeighbors 4
        self.detectedFaces = faceCascade.detectMultiScale( self.transformToGrayscale, 1.1, 4)
        self.detectedEyes = eyesCascade.detectMultiScale( self.transformToGrayscale, 1.1, 4)
        
        # use the coordinates to draw a rectangle on the detected faces and eyes
        self.countFaces = len( self.detectedFaces) 
        self.countEyes = len( self.detectedEyes) 
        self.loopIteration = 0
        while True:
            if self.loopIteration < self.countFaces:
                (x, y, w, h) = self.detectedFaces[self.loopIteration]
                cv2.rectangle( self.videoFrame, (x, y), (x+w, y+h), ( 0, 255, 0 ), 2)
            if self.loopIteration < self.countEyes:
                (x, y, w, h) = self.detectedEyes[self.loopIteration]
                cv2.rectangle( self.videoFrame,(x,y),(x+w,y+h),(255, 0, 0),2)
            self.loopIteration += 1
            if self.loopIteration > self.countFaces and self.loopIteration > self.countEyes:
                break

        # An alternate solution?
        # But this uses two loops so it might be a lot slower
        #for (x, y, w, h) in self.detectedFaces:
        #    cv2.rectangle( self.videoFrame, (x, y), (x+w, y+h), ( 0, 255, 0 ), 2)
        #for (x,y,w,h) in self.detectedEyes:
        #    cv2.rectangle( self.videoFrame,(x,y),(x+w,y+h),(255, 0, 0),2)
        

        _, outputFrame = cv2.imencode( ".jpg", self.videoFrame )
        return outputFrame.tobytes()

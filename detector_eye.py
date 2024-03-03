import cv2
import numpy as np

# Load Haar Cascade classifiers for face and eye detection
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeClassif = cv2.CascadeClassifier('haarcascade_eye.xml')

# Read the input image
image = cv2.imread('retrato.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = faceClassif.detectMultiScale(gray,
        scaleFactor=1.01,
        minNeighbors=5,
        minSize=(30,30),
        maxSize=(200,200))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Extract the region of interest (ROI) for eyes within the detected face region
    roi_gray = gray[y:y+h, x:x+w]
    
    # Perform eye detection within the face region
    eyes = eyeClassif.detectMultiScale(roi_gray)
    
    # Draw rectangles around the detected eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)

# Display the image with the detected faces and eyes
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


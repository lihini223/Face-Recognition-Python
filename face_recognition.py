import cv2
import sys

# Get user inputs
image_path = sys.argv[1]
cascade_path = sys.argv[2]

# Initialize the cascade
face_cascade = cv2.CascadeClassifier(cascade_path)

# Read the image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect face
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print(f'Found {(len(faces))} faces!')

# Draw a bounding box around the faces
for (x, y, w, h) in faces:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Faces Found", image)

cv2.waitKey(0)

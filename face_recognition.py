import cv2

# Get user inputs
image_path = sys.argv[1]
cascade_path = sys.argv[2]

# Initialize the cascade
face_cascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect face
faces = faceCascade.detectMultiScale(
  gray,
  scaleFactor = 1.1,
  minNeighbors = 5,
  minSize = (30, 30)
  flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

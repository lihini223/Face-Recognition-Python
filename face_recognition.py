import cv2

# Get user inputs
image_path = sys.argv[1]
cascade_path = sys.argv[2]

# Initialize the cascade
face_cascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


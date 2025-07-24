import cv2
from deepface import DeepFace

# Load Haarcascade face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region for analysis
        face_roi = img[y:y + h, x:x + w]

        try:
            # Use DeepFace to analyze the emotion
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Get the dominant emotion
            emotion = analysis[0]['dominant_emotion']

            # Display emotion on the frame
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except:
            pass  # In case DeepFace fails, continue

    # Show the video feed
    cv2.imshow('Face & Emotion Detection', img)

    # Exit if 'Esc' key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

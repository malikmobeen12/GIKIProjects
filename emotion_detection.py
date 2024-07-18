import cv2
from keras.models import load_model
import numpy as np


def emotion_detection():

    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    model = load_model("emotion.h5")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Process each detected face
        for x, y, w, h in faces:
            # Extract face region
            face = gray[y : y + h, x : x + w]

            # Resize and preprocess the face image
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
            face = face / 255.0  # Normalize pixel values

            # Predict emotion using the pre-trained model
            emotions = model.predict(face)
            emotion_label = emotion_labels[np.argmax(emotions)]

            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display predicted emotion label
            cv2.putText(
                frame,
                "Emotion: {}".format(emotion_label),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        # Display the annotated frame
        cv2.imshow("Emotion Detection", frame)

        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

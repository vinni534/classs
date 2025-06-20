import face_recognition
import cv2

# Load a sample image and learn how to recognize it.
known_image = face_recognition.load_image_file("known.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Load an unknown image to recognize faces in
unknown_image = face_recognition.load_image_file("unknown.jpg")
unknown_face_locations = face_recognition.face_locations(unknown_image)
unknown_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)

# Initialize OpenCV window
for (top, right, bottom, left), face_encoding in zip(unknown_face_locations, unknown_encodings):
    matches = face_recognition.compare_faces([known_encoding], face_encoding)
    
    name = "Unknown"
    if matches[0]:
        name = "Known Person"

    # Draw box around the face
    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
    # Label the face
    cv2.putText(unknown_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Convert color from BGR to RGB before showing
rgb_image = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

# Show the result
cv2.imshow('Face Recognition', rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

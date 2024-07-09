import cv2
# to install face_recognition install first cmake and dlib 
import face_recognition
import time


# Load known faces and their names
known_face_encodings = []
known_face_names = []

known_face1_img = face_recognition.load_image_file("faces/piolopascual.jpg")
known_face2_img = face_recognition.load_image_file("faces/jm.png")

known_face1_encoding = face_recognition.face_encodings(known_face1_img)[0]
known_face2_encoding = face_recognition.face_encodings(known_face2_img)[0]


known_face_encodings.append(known_face1_encoding)
known_face_encodings.append(known_face2_encoding)

known_face_names.append("Piolo Pascual")
known_face_names.append("Jm")

# Initialize webcam
start = time.time()
cap = cv2.VideoCapture(0)
end = time.time()
print(end-start)



while True:
    start = time.time()
    success, img = cap.read()

    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(img)
    face_encoding = face_recognition.face_encodings(img,face_locations)

    # Loop through each face found in the frame 
    for (top,right,bottom,left), face_encoding in zip(face_locations,face_encoding):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

         # Draw a box  around  the face and label with  name
        cv2.rectangle(img, (left,top), (right,bottom), (0,0,255), 2)   
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
    # Display the resulting frame
    cv2.imshow("Image", img)
    
    end = time.time()
    print(end-start)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

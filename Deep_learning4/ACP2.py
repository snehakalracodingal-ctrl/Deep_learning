
import face_recognition
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

from google.colab import files

from google.colab import files

print("Upload Elon Musk image")
uploaded = files.upload()
elon_file = list(uploaded.keys())[0]

print("Upload Donald Trump image")
uploaded = files.upload()
trump_file = list(uploaded.keys())[0]

print("Upload Jeff Bezos image")
uploaded = files.upload()
jeff_file = list(uploaded.keys())[0]

print("Upload unknown image")
uploaded = files.upload()
unknown_file = list(uploaded.keys())[0]

face_1 = face_recognition.load_image_file(elon_file)
face_1_encoding = face_recognition.face_encodings(face_1)[0]

face_2 = face_recognition.load_image_file(trump_file)
face_2_encoding = face_recognition.face_encodings(face_2)[0]

face_3 = face_recognition.load_image_file(jeff_file)
face_3_encoding = face_recognition.face_encodings(face_3)[0]


known_face_encodings = [
    face_1_encoding,
    face_2_encoding,
    face_3_encoding
]

known_face_names = [
    "Elon Musk",
    "Donald Trump",
    "Jeff Bezos"
]

file_name = unknown_file
unknown_img = face_recognition.load_image_file(file_name)
unknown_img_2draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_img)
face_encodings = face_recognition.face_encodings(unknown_img, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"  # Default name if no match is found

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

   
    if True in matches:
        best_match_index = np.argmin(face_distances)
        name = known_face_names[best_match_index]  # Get the name of the closest match


    cv2.rectangle(unknown_img_2draw, (left, top), (right, bottom), (0, 0, 255), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(unknown_img_2draw, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

cv2_imshow(unknown_img_2draw)

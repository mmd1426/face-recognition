# Import libraries
import pickle
import cv2
import face_recognition

# Drawing border in around face
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


# Load model
data = pickle.load(open(r'/src/model/model.pkl', 'rb'))

encodings = data['encodings']
names = data['names']

cascade = cv2.CascadeClassifier(r'/src/haarcascade/haarcascade_frontalface_default.xml')

# Load webcam
cap = cv2.VideoCapture(0)

while True:

    # Read frame
    ret,frame = cap.read()

    # Detect faces in frame webcam
    face_detection = cascade.detectMultiScale(frame,1.3, 15)

    # Actions in frame
    for (x,y,w,h) in face_detection:
        draw_border(frame,(x,y),(x+w,y+h),(0,255,0),2,15,5)
        face = frame[y:y+h,x:x+w]
        img_enc = face_recognition.face_encodings(face)
        if len(img_enc) > 0:
            results = face_recognition.compare_faces(encodings, img_enc[0])
            if True in results:
                i = results.index(True)
                name = names[i]
                name = str(name).split('_')
                name = ' '.join(name)
                cv2.putText(frame, name, (x+25,y-20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(frame, 'Unknow', (x+25,y-20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)




    cv2.imshow('Project',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
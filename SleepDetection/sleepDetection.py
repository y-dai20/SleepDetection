import cv2
import os.path
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np

path = os.getcwd()

class SleepDetection:
    
    def __init__(self, img):

        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        path = os.getcwd()

        self.cascade = cv2.CascadeClassifier(
            path + '/haarcascades/haarcascade_frontalface_alt2.xml'
        )

        self.eye_cascade = cv2.CascadeClassifier(
            path + '/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
        )

        self.face_parts_detector = dlib.shape_predictor(
            path + '/dlib/shape_predictor_68_face_landmarks.dat'
        )

    def calc_ear(self, eye):
        # A = distance.euclidean(eye[1], eye[5])
        # B = distance.euclidean(eye[2], eye[4])
        # C = distance.euclidean(eye[0], eye[3])
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        eye_ear = (A + B) / (2.0 * C)
        return round(eye_ear, 3)

    #顔部分の情報を検出
    def detect_faces(self):
        facerect = self.cascade.detectMultiScale(
            self.gray,
            scaleFactor=1.11,
            minNeighbors=2,
            minSize=(30, 30)
        )
  
        faces = []
        if len(facerect) != 0:
            for x, y, w, h in facerect:
                # 顔の部分
                faces.append({'x': x, 'y': y, 'w': w, 'h': h})

        return faces

    def detect_face_parts(self, face):
        around_face = dlib.rectangle(
            face["x"], 
            face["y"], 
            face["x"]+face["w"], 
            face["y"]+face["h"]
        )
        face_parts = self.face_parts_detector(self.img, around_face)
        face_parts = face_utils.shape_to_np(face_parts)

        # for i, ((x, y)) in enumerate(face_parts[:]):
        #     cv2.circle(self.img, (x, y), 1, (0, 255, 0), -1)
        #     cv2.putText(self.img, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
        left_eye_ear = self.calc_ear(face_parts[42:48])
        right_eye_ear = self.calc_ear(face_parts[36:42])

        print(left_eye_ear, right_eye_ear)
        if (left_eye_ear + right_eye_ear) < 0.50:
            cv2.putText(self.img, "Oh",
                (face["x"],face["y"]), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, 1)
        else:
            cv2.putText(self.img, "Good",
                (face["x"],face["y"]), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 3, 1)

    # 目を閉じているか
    def is_closed_eyes(self, face):
        # 顔の部分
        face_x = face['x']
        face_y = face['y']
        face_w = face['w']
        face_h = face['h']

        # 顔の部分から目の近傍を取る
        eyes_gray = self.gray[face_y: face_y + face_h, face_x: face_x + face_w]

        self.eyes = self.eye_cascade.detectMultiScale(
            eyes_gray,
            scaleFactor=1.11,
            minNeighbors=1,
            minSize=(1, 1)
        )

        # どちらかの目が開いていればOK
        return len(self.eyes) == 0
    
    def draw_eye_rectangle(self, face):
        for ex, ey, ew, eh in self.eyes:
            cv2.rectangle(
                self.img, 
                (face['x'] + ex, face['y'] + ey), 
                (face['x'] + ex + ew, face['y'] + ey + eh), 
                (255, 255, 0), 
                2
            )

    def draw_face_rectangle(self, face):
        cv2.rectangle(
            self.img, 
            (face['x'], face['y']), 
            (face['x'] + face['w'], face['y'] + face['h']), 
            (255, 0, 0), 
            2
        )
    
def main():
    #変更可能性あり
    img = cv2.imread(path + "/SleepDetection/image/they.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sd = SleepDetection(img)
    faces = sd.detect_faces()
    # print(faces)
    for i in range(len(faces)):
        sd.draw_face_rectangle(faces[i])
        sd.detect_face_parts(faces[i])
        if sd.is_closed_eyes(faces[i]) is False:
            sd.draw_eye_rectangle(faces[i])

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    main()
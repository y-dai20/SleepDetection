import cv2
import matplotlib.pyplot as plt

class SleepDetection:
    
    def __init__(self, img):

        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.cascade = cv2.CascadeClassifier(
            '/Users/yamada/Documents/TUS/3年/後期/応用情報工学演習/谷口研/課題/gw/SleepDetection/haarcascades/haarcascade_frontalface_alt2.xml'
        )
        # leftとrightは逆転する
        # self.left_eye_cascade = cv2.CascadeClassifier(
        #     '/Users/yamada/Documents/TUS/3年/後期/応用情報工学演習/谷口研/課題/gw/SleepDetection/haarcascades/haarcascade_righteye_2splits.xml'
        # )
        # self.right_eye_cascade = cv2.CascadeClassifier(
        #     '/Users/yamada/Documents/TUS/3年/後期/応用情報工学演習/谷口研/課題/gw/SleepDetection/haarcascades/haarcascade_lefteye_2splits.xml'
        # )

        self.eye_cascade = cv2.CascadeClassifier(
            '/Users/yamada/Documents/TUS/3年/後期/応用情報工学演習/谷口研/課題/gw/SleepDetection/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
        )
    #顔部分の情報を検出
    def detect_face_parts(self):
        facerect = self.cascade.detectMultiScale(
            self.gray,
            scaleFactor=1.11,
            minNeighbors=3,
            minSize=(100, 100)
        )

        if len(facerect) != 0:
            for x, y, w, h in facerect:
                # 顔の部分
                return {'x': x, 'y': y, 'w': w, 'h': h}

        return {}

    # 目を閉じているか
    def is_closed_eyes(self, face_parts):
        # 顔の部分
        face_x = face_parts['x']
        face_y = face_parts['y']
        face_w = face_parts['w']
        face_h = face_parts['h']

        # 顔の部分から目の近傍を取る
        eyes_gray = self.gray[face_y: face_y + int(face_h/2), face_x: face_x + face_w]
        # cv2.imshow('face', eyes_gray)

        min_size = (8, 8)  # 調整いるかも

        ''' 目の検出
        眼鏡をかけている場合、精度は低くなる。
        PCのスペックが良ければ、haarcascade_eye_tree_eyeglasses.xmlを使ったほうがよい。
        '''
        # left_eye = self.left_eye_cascade.detectMultiScale(
        #     eyes_gray,
        #     scaleFactor=1.11,
        #     minNeighbors=3,
        #     minSize=min_size
        # )
        # right_eye = self.right_eye_cascade.detectMultiScale(
        #     eyes_gray,
        #     scaleFactor=1.11,
        #     minNeighbors=3,
        #     minSize=min_size
        # )
        eyes = self.eye_cascade.detectMultiScale(
            eyes_gray,
            scaleFactor=1.11,
            minNeighbors=3,
            minSize=min_size
        )

        ''' left_eye, right_eye
        [[116  40  36  36] [34  40  40  40]] => 開いている
        [[34 40 41 41]] => 閉じている
        [] => 未検出
        '''
        # for ex,ey,ew,eh in right_eye:
        #     cv2.rectangle(self.img, (face_x + ex, face_y + ey), (face_x + ex + ew, face_y + ey + eh), (255, 255, 0), 1)
        # for ex,ey,ew,eh in left_eye:
        #     cv2.rectangle(self.img, (face_x + ex, face_y + ey), (face_x + ex + ew, face_y + ey + eh), (255, 255, 0), 1)

        for ex, ey, ew, eh in eyes:
            cv2.rectangle(self.img, (face_x + ex, face_y + ey), (face_x + ex + ew, face_y + ey + eh), (255, 255, 0), 2)
        # 片目だけ閉じても駄目にしたい場合(これだと結構厳しい(精度悪い？)判定になる)
        # return len(left_eye) <= 1 or len(right_eye) <= 1

        # どちらかの目が開いていればOK
        return len(eyes) == 0

    #四角で囲む
    def output_image(self):
        return True
def main():
    #変更可能性あり
    img = cv2.imread("/Users/yamada/Documents/TUS/3年/後期/応用情報工学演習/谷口研/課題/gw/SleepDetection/yamada.jpg")

    sd = SleepDetection(img)
    face_parts = sd.detect_face_parts()
    if len(face_parts) != 0 and sd.is_closed_eyes(face_parts) is False:
        x = face_parts['x']
        y = face_parts['y']
        w = face_parts['w']
        h = face_parts['h']

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main()
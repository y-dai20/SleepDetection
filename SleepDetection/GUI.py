from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
from pathlib import Path
import cv2
from SleepDetection import SleepDetection
screenSize = 500

class MyWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(screenSize,screenSize)
        self.statusBar()
        
        self.image = QLabel(self)
        openFile = QAction('&Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.getImage)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)

        self.setWindowTitle('File dialog')
        self.show()

    def getImage(self):
        home_dir = str(Path.home())
        fname = QFileDialog.getOpenFileName(self, 'Open file', home_dir)

        if fname[0]:
            self.processing(fname[0])

        return []

    def processing(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        maxSize = max(img.shape[0], img.shape[1])
        img = cv2.resize(img, (int(img.shape[1] / maxSize * screenSize), int(img.shape[0] / maxSize * screenSize)))
        
        sd = SleepDetection(img)
        faces = sd.detect_faces()
        
        for i in range(len(faces)):
            sd.draw_face_rectangle(faces[i])
            sd.detect_face_parts(faces[i])
            if sd.is_closed_eyes(faces[i]) is False:
                sd.draw_eye_rectangle(faces[i])

            self.showDialog(img)

    def showDialog(self, img):
        h, w, d = img.shape
        qimg = QImage(img.data, w, h, d * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image.setPixmap(pixmap)

        hbox = QHBoxLayout()
        hbox.addWidget(self.image)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        container = QWidget()
        container.setLayout(vbox)
        self.setCentralWidget(container)
        self.show()

def main():
    app = QApplication(sys.argv)
    mw = MyWindow()
    app.exec_()

if __name__ == '__main__':
    main()
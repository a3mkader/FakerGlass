import cv2
import numpy as np
import face_recognition as fr
import os
import time
import gtts
import pickle
from pygame import mixer

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimensions = (width, height)
    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

class FaceDetection:
    def __init__(self):
        # variables
        self.path = './People'
        self.images = []
        self.names = []
        self.state = True


    def train(self):
        mylist = os.listdir(self.path)

        for img in mylist:
            currentimg = cv2.imread(f'{self.path}/{img}')
            self.images.append(currentimg)
            self.names.append(os.path.splitext(img)[0])

        names_f=open("./data/names.pkl","wb")
        pickle.dump(self.names,names_f)
        names_f.close()

        for name in self.names:
            tts = gtts.gTTS(f'{name}', lang="en")
            tts.save(f"./Sound/{name}.mp3")

        tts = gtts.gTTS("unknown", lang="en")
        tts.save("./Sound/unknown.mp3")

        def encoding(images: list):
            encodinglist = []
            for img in images:
                img = resize(img, 0.50)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                enc = fr.face_encodings(img)[0]
                encodinglist.append(enc)
            return encodinglist


        enc_list = encoding(self.images)
        enc_f=open("./data/enc.pkl","wb")
        pickle.dump(enc_list,enc_f)
        enc_f.close()

    def test(self):
        while self.state:
            with open('./data/names.pkl', 'rb') as f:
                names_f_ = pickle.load(f)
            with open('./data/enc.pkl', 'rb') as f:
                enc_f_ = pickle.load(f)

            os.chdir('./taken_images')
            camera = cv2.VideoCapture(0)
            for i in range(1):
                return_value, image = camera.read()
                cv2.imwrite('0' + str(i) + '.png', image)
            del (camera)
            os.chdir('..')
            img = cv2.imread('./taken_images/00.png')
            frames = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            frames = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faces = fr.face_locations(frames)
            encode_in_frame = fr.face_encodings(frames, faces)

            for encode, loc in zip(encode_in_frame, faces):
                matches = fr.compare_faces(enc_f_, encode)
                facedis = fr.face_distance(enc_f_, encode)
                matchIndex = np.argmin(facedis)
                if matches[matchIndex]:
                    name = names_f_[matchIndex]
                    mixer.init()
                    mixer.music.load(f'Sound/{name}.mp3')
                    mixer.music.play()

                else:
                    name = 'unknown'
                    mixer.init()
                    mixer.music.load(f'Sound/{name}.mp3')
                    mixer.music.play()
            time.sleep(2)
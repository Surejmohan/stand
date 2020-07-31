from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template,redirect,url_for
import threading
import datetime
import time
from keras.models import load_model
import cv2
import numpy as np
import time

outputFrame = None
lock = threading.Lock()


model = load_model('model-017.model')
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}
app = Flask(__name__)

def Main():
    FaceDetection()
    time.sleep(5.0)
    return Main()
    


def FaceDetection():
    
    source=cv2.VideoCapture(0)
    

    while(True):

        ret,img=source.read()
        if not ret:
            break
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_clsfr.detectMultiScale(gray,1.3,5)

        if(len(faces) == 1):
            print("Face Detected \n")
            print("Please Wait")
            source.release()
            time.sleep(1.0)
            k = MaskDetection()
            if(k == 0):
                print("Quit")
                cv2.destroyAllWindows()
                return 1



@app.route("/")
def index():
    
    t = threading.Thread(target=Main)
    
    t.daemon = True
    t.start()
    return render_template("index.html")

def MaskDetection():
    
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    global outputFrame, lock ,string

    while True:
        
        img=vs.read()
        
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_clsfr.detectMultiScale(gray,1.3,5)  

        for (x,y,w,h) in faces:

            face_img=gray[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            result=model.predict(reshaped)

            label=np.argmax(result,axis=1)[0]
            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
            if label == 0:
                cv2.putText(img,"MASK FOUND", (x+10, y-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

            if label == 1:
                cv2.putText(img,"NO MASK FOUND", (x+10, y-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        with lock:
            outputFrame = img.copy()
    
        
def generate():
    
    global outputFrame, lock

    while True:
        with lock:
        
            if outputFrame is None:
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():

    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    import random, threading, webbrowser
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    threading.Timer(1.25, lambda: webbrowser.open(url) ).start()
    
    app.run(port=port, debug=False)


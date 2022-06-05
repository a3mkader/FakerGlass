from flask import Flask, render_template
from faceDetection import FaceDetection
import os

app= Flask(__name__)
fd = FaceDetection()
@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/train')
def train():
    fd.__init__()
    fd.train()
    return render_template('train.html')

@app.route('/test')
def test():
    fd.state= True
    fd.test()
    return render_template('result.html')

@app.route('/stop')
def stop():
    fd.state= False
    return render_template('result.html')



app.run(debug=True)

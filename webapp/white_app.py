from flask import Flask, jsonify, render_template, request
from white_position import *
import cv2

app = Flask(__name__)

@app.route('/',methods = ['POST'])

def white_app():
    if request.files.get('file'):
        file = request.files.get('file')
        file.save('./picture/'+file.filename)
        image = cv2.imread('./picture/'+file.filename)
        white = White()
        widthrate,lengthrate,angle = white.white_process(image)
        return "水平位置：{:.2f} 竖直位置：{:.2f} 角度：{:.2f}".format(widthrate,lengthrate,angle)
    else:
        return "Error"
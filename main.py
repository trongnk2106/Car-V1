"""
        * depth_image: set takeDepth = True,  depth image will return when 'send_control' sent)
        + sendBack_angle [-25, 25]  
        + sendBack_Speed [-150, 150] 
"""
import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import cv2
from PIL import Image
from flask import Flask
from io import BytesIO

# My import 
from net import Net
from src.parameters import Parameters
from src import util
import copy

import torch
import time

# My setup

net = Net()
p = Parameters()
warning = []
SetStatusObjs = []
StatusLines = []
StatusBoxes = []

#Global variable
MAX_SPEED = 30
MAX_ANGLE = 25
#Tốc độ thời điểm ban đầu
speed_limit = MAX_SPEED
MIN_SPEED = 10

#init our model and image array as empty


#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#registering event handler for the server
flag_stop = False

@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        steering_angle = float(data["steering_angle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        
        
        sendBack_angle = 0
        sendBack_Speed = 70
        try:
        #   #------------------------------------------  Work space  ----------------------------------------------#
            
            prevTime = time.time()
            
            ###Lane 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(512,256))
        
            x, y = net.predict(image)
            fits = np.array([np.polyfit(_y, _x, 1) if len(_x) < 5  else  np.polyfit(_y, _x, 2) for _x, _y in zip(x, y)])
            fits = np.array([np.polyfit(_y, _x, 1) for _x, _y in zip(x, y)])
            
            fits = util.adjust_fits(fits)
            
            StatusLines.append(len(fits))
            if len(StatusLines) > 8:
                StatusLines = StatusLines[-8:]
            

            curTime = time.time()
            image_lane = net.get_image_points()
            sendBack_angle = util.get_steer_angle(fits)
            


            sec = curTime - prevTime
            fps = 1/(sec)
            s = "FPS : "+ str(fps)
            print(s)
            net.get_mask_lane(fits)
            image_lane = net.get_image_lane()
            cv2.putText(image_lane, s, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv2.imshow("image", image)
            cv2.waitKey(1)

            
            #------------------------------------------------------------------------------------------------------#
            print('{} : {}'.format(sendBack_angle, sendBack_Speed))
            if speed > MAX_SPEED:
                sendBack_Speed = 5
            if speed < 10:
                sendBack_Speed = 100
            send_control(sendBack_angle, sendBack_Speed)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
        },
        skip_sid=True)


if __name__ == '__main__':
    
    config_file = "/home/trong/code/Car/darknet/cfg/yolo-tiny-v4-custom.cfg"
    weights = "/home/trong/code/Car/darknet/models/yolo-tiny-v4-custom_last.weights"
    data_file = "/home/trong/code/Car/darknet/cfg/tiny-traffic-sign.data"
    
    #-----------------------------------  Setup  ------------------------------------------#
    print('I am loading model right now, pls wait a minute')
    net.load_model(34,"0.7828")

    
    #--------------------------------------------------------------------------------------#
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

# Import socket module
import socket
import cv2
import numpy as np

import copy

import torch
import time

# My setup
from src import util
from net import Net
net = Net()
#p = Parameters()
#warning = []
SetStatusObjs = []
StatusLines = []
StatusBoxes = []

#Global variable
MAX_SPEED = 35
MAX_ANGLE = 25
#Tốc độ thời điểm ban đầu
speed_limit = MAX_SPEED
MIN_SPEED = 10

prevTime = time.time()


global sendBack_angle, sendBack_Speed, current_speed, current_angle
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
PORT = 54321
# connect to the server on local computer
net.load_model(49,"0.2270")
s.connect(('host.docker.internal', PORT))


def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed


if __name__ == "__main__":
    print('I am loading model right now, pls wait a minute')
    

    try:
        while True:

            """
            - Chương trình đưa cho bạn 1 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe

            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)

                    + sendBack_Speed (tốc độ điều khiển): [-150, 150]
                        NOTE: (âm là lùi, dương là tiến)
            """
            '''message_getState = bytes(f'{sendBack_angle} {sendBack_Speed}', 'utf-8')
            #message_getState = bytes("0", "utf-8")
            s.sendall(message_getState)
            state_date = s.recv(100)

            try:
                current_speed, current_angle = state_date.decode(
                    "utf-8"
                    ).split(' ')
            except Exception as er:
                print(er)
                pass
            '''
            message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            data = s.recv(100000)

            try:
                image = cv2.imdecode(
                    np.frombuffer(
                        data,
                        np.uint8
                        ), -1
                    )

                #print(current_speed, current_angle)
                #print(image.shape)

                # your process here
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image,(512,256))
                #pro_img = cv2.inRange(image, (190,190,190), (255,255,255))
                #cv2.imshow('processed image',image)
                # your process here

                x, y = net.predict(image)
                fits = np.array([np.polyfit(_y, _x, 1) if len(_x) < 5  else  np.polyfit(_y, _x, 2) for _x, _y in zip(x, y)])
                fits = np.array([np.polyfit(_y, _x, 1) for _x, _y in zip(x, y)])
                
                fits = util.adjust_fits(fits)
                print("fits: ", fits.shape[0])
                sendBack_angle = util.get_steer_angle(fits, current_speed)
                curTime = time.time()
                sec = curTime - prevTime
                fps = 1/(sec)
                string = "FPS : "+ str(fps)
                #print("FPS: ", string)

                net.get_mask_lane(fits)
                #image_lane = net.get_image_lane()
                image_points = net.get_image_points()
                #cv2.imshow("image", image_points)
                
                #cv2.imshow("image", image)
                #key = cv2.waitKey(1)
                if sendBack_Speed > MAX_SPEED:
                    sendBack_Speed = 5
                if sendBack_Speed < 10:
                    sendBack_Speed = 35
                # if (sendBack_angle >= 5 or sendBack_angle <= -5):
                #     sendBack_Speed = 5

                # cv2.imshow("IMG", image)
                cv2.waitKey(1)
                # Control(angle, speed)
                #Control(sendBack_angle, sendBack_Speed)

            except Exception as er:
                print(er)
                pass

    finally:
        print('closing socket')
        s.close()

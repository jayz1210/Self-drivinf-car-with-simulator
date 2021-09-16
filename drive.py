from flask import Flask
import eventlet
import socketio
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow import keras
import cv2

sio = socketio.Server()


app = Flask(__name__)
speed_limit = 10

def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3),0)
    img = cv2.resize(img,(200,66))
    img= img/255

    return img

    return img
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        image1 = cv2.resize(image_array, (66,200))
        steering_angle = float(model.predict(image1[None, :, :, :], batch_size=1))
        min_speed = 12
        max_speed = 24
        if float(speed) < min_speed:
            throttle = 1.0/10
        elif float(speed) > max_speed:
            throttle = -1.0
        else:
            throttle = 0.2

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        #if args.image_folder != '':
        #    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        #    image_filename = os.path.join(args.image_folder, timestamp)
        #    image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid,environ):
    print('connect')
    send_control(0,0)

def send_control(steering_angle,throttle):
    sio.emit('steer',data ={
    'steering_angle':steering_angle.__str__(),
    'throttle':throttle.__str__()
    })
#send_control(0,1)


if __name__=='__main__':
    model = load_model('model.h5')
    print("Original model:", model)
    app = socketio.Middleware(sio,app)
    eventlet.wsgi.server(eventlet.listen(('',4567)),app)

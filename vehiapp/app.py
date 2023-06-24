import cv2
import easyocr
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import tensorflow as tf
from tensorflow import keras
from keras.utils import CustomObjectScope
from matplotlib import pyplot as plt
import numpy as np
import imutils
from PIL import Image, ImageOps
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from flask import Flask,request, jsonify
from configparser import Interpolation
import tempfile

app = Flask(__name__)

Vehicle_model = tf.keras.models.load_model("model_vehiscan.h5",compile=False)
haarcascade="haarcascade_russian_plate_number.xml"
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()


cred = credentials.Certificate("vehiscan_firebase.json")
firebase_admin.initialize_app(cred)
db=firestore.client()
    
def detection():
    H = 224
    W = 224
    image = cv2.imread('temp.jpg', cv2.IMREAD_COLOR)
    height, width, _ = image.shape
    resized_image = cv2.resize(image, (W, H))
    image_array = np.asarray(resized_image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    predicted_vehicle = Vehicle_model.predict(data)
    index = np.argmax(predicted_vehicle)
    if index==0:
        vehicle_type="auto"
    elif index==1:
        vehicle_type="bike"
    elif index==2:
        vehicle_type="bus"
    elif index==3:
        vehicle_type="car"

    min_area=500
    count=0
    plate_cascade=cv2.CascadeClassifier(haarcascade)
    img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    plates=plate_cascade.detectMultiScale(img_gray,1.1,4)
    
    vehicle_number='0'
    for(x,y,w,h) in plates:
        area=w*h
        if area>min_area:
            img_roi=image[y:y+h,x:x+w]#cropping
            reader=easyocr.Reader(['en'],gpu=False, verbose=False)
            result=reader.readtext(img_roi,detail=1,paragraph=False)
            vehicle_number=(result[0][-2]).replace(" ","")
            
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image_d = Image.open('temp.jpg').convert("RGB")    # Convert the NumPy array to a PIL Image object
    size = (224, 224)
    image = ImageOps.fit(image_d, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    vehicle_color=class_name
    
    vehicle_ref=db.collection('vehicles').document(vehicle_number)
    vehicle_doc=vehicle_ref.get()
    if vehicle_doc.exists:
        db_vehicle_type=vehicle_doc.get('type')
        db_vehicle_color=vehicle_doc.get('color')
        result={
            'detected_color':vehicle_color,
            'detected_number':vehicle_number,
            'detected_type':vehicle_type,
            'original_type':db_vehicle_type,
            'original_color':db_vehicle_color,
            'status':'true'
        }
    else:
        result={
            'status':'false'
        }
    return result


@app.route('/process', methods=['POST'])
def process():
    image= request.files['image']
    
    # with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
    image.save('temp.jpg')
    result =jsonify(detection())
    return result

@app.route('/', methods=['GET'])
def home():
    return "HELLO"

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
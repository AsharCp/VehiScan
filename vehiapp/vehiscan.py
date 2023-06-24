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
H = 224 # height
W = 224  # width
# Read image 
image = cv2.imread("vehicleimage.jpg", cv2.IMREAD_COLOR)
# image = Image.open("vehicleimage.jpg").convert("RGB")
# taking the width and height of the originl image
height, width, _ = image.shape
# resizing the image to the expected shape of (None, 224, 224, 3)
resized_image = cv2.resize(image, (W, H))
# Load vehicle type detection model
# <-----------------------Vehicle type detection -------------------->
Vehicle_model = tf.keras.models.load_model("model_vehiscan.h5",compile=False)
image_array = np.asarray(resized_image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Load the image into the array
data[0] = normalized_image_array
predicted_vehicle = Vehicle_model.predict(data)
# resized_image = np.expand_dims(resized_image, axis=0)
index = np.argmax(predicted_vehicle)
if index==0:
  vehicle_type="auto"
elif index==1:
  vehicle_type="bike"
elif index==2:
  vehicle_type="bus"
elif index==3:
  vehicle_type="car"
# Print the vehicle type result
# print(vehicle_type, end="")
# img = cv2.imread('vehicleimage.jpg')

# <---------------Vehicle_model--------------->

haarcascade="haarcascade_russian_plate_number.xml"
# img = cv2.imread("/content/drive/MyDrive/Vehi_Scan/vehicleimage.jpg")
min_area=500
count=0
plate_cascade=cv2.CascadeClassifier(haarcascade)
img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plates=plate_cascade.detectMultiScale(img_gray,1.1,4)

from configparser import Interpolation
for(x,y,w,h) in plates:
    area=w*h
    if area>min_area:
        # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.putText(image,"Number Plate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),2)

        img_roi=image[y:y+h,x:x+w]#cropping
        # cv2_imshow(img_roi)
        # cv2_imshow(img_roi)
        # cv2.waitKey(0)
        # cv2.imwrite("plates/result_image_1.png",img_roi)
        # gray=cv2.cvtColor(img_roi,cv2.COLOR_BGR2GRAY)
        # gray=cv2.resize(gray,None,fx=3,fy=3,Interpolation=cv2.INTER_CUBIC)
        # gray=cv2.cvtColor(box,cv2.COLOR_BGR2GRAY)
        # blur=cv2.GaussianBlur(gray,(5,5),0)
        # ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        # rect_kern=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        # dilation=cv2.dilate(thresh,rect_kern,iterations=1)
        # contours,hierarchy=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # sorted_contours=sorted(contours,key=lambda ctr:cv2.boundingRect(ctr)[0])


        # cv2_imshow(gray)
        reader=easyocr.Reader(['en'],gpu=False, verbose=False)
        result=reader.readtext(img_roi,detail=1,paragraph=False)
        # pipeline = keras_ocr.pipeline.Pipeline()
        # result = pipeline.recognize([gray])
        # pd.DataFrame(result[0])
        vehicle_number=(result[0][-2]).replace(" ","")
        # The number plate is stored in variable number_text
        # print(number_text)
        # cv2.waitKey(0)

# <------------------end----------------------------->

# <------------------Vehicle number detection------------>
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
# bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
# edged = cv2.Canny(bfilter, 30, 200) #Edge detection
# # plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
# keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(keypoints)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
# location = None
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 10, True)
#     if len(approx) == 4:
#         location = approx
#         break
# mask = np.zeros(gray.shape, np.uint8)
# new_image = cv2.drawContours(mask, [location], 0,255, -1)
# new_image = cv2.bitwise_and(image, image, mask=mask)
# plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
# (x,y) = np.where(mask==255)
# (x1, y1) = (np.min(x), np.min(y))
# (x2, y2) = (np.max(x), np.max(y))
# cropped_image = gray[x1:x2+1, y1:y2+1]
# # easyocr
# reader = easyocr.Reader(['en'], verbose=False)
# result = reader.readtext(cropped_image)
# print(result)
# vehicle_number=(result[0][-2]).replace(" ","")
# print vehcile cnumber
# print(vehicle_number)
# <-------------Detecting vehicle color------------->
np.set_printoptions(suppress=True)
# Load color detection model
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image_d = Image.open("vehicleimage.jpg").convert("RGB")
# Convert the NumPy array to a PIL Image object
# image = Image.fromarray(image_array)
# image = image.convert("RGB")
# Resize and crop the image
size = (224, 224)
# image = ImageOps.fit(image, size, Image.ANTIALIAS)
image = ImageOps.fit(image_d, size, Image.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data[0] = normalized_image_array
prediction = model.predict(data)
index = np.argmax(prediction)
# print(index)
class_name = class_names[index]
# confidence_score = prediction[0][index]
vehicle_color=class_name
# print("Confidence Score:", confidence_score)
# Print vehicle color
# print(vehicle_color, end="")
# Firebase connection 
# <--------------Firebase Connection--------->
if not firebase_admin._apps:
  cred = credentials.Certificate("vehiscan_firebase.json")
  firebase_admin.initialize_app(cred)
  db=firestore.client()
  vehicle_ref=db.collection('vehicles').document(vehicle_number)
  vehicle_doc=vehicle_ref.get()
if vehicle_doc.exists:
  db_vehicle_type=vehicle_doc.get('type')
  db_vehicle_color=vehicle_doc.get('color')
  # print(db_vehicle_type)
  # print(db_vehicle_color)
  # print(vehicle_color)
  if (vehicle_type==db_vehicle_type) and (vehicle_color==db_vehicle_color):
    vehicle_result='Real Vehicle'
    print("The vehicle is Real!")
  elif (vehicle_type==db_vehicle_type) and (vehicle_color!=db_vehicle_color):
    vehicle_result='Fake vehicle-original color is'+db_vehicle_color
    print("The vehicle is Fake.The color is altered.The original color is",db_vehicle_color)
    # print(vehicle_color)
    # print(db_vehicle_color)
  elif (vehicle_type!=db_vehicle_type) and (vehicle_color==db_vehicle_color):
    vehicle_result='Fake vehicle-It is the vehicle number of a'+db_vehicle_type
    print("The vehicle is Fake.The number plate is altered.It is the vehicle number of a",db_vehicle_type)
  elif (vehicle_type!=db_vehicle_type) and (vehicle_color!=db_vehicle_color):
    vehicle_result='Fake vehicle-original color is'+db_vehicle_color+' and the original vehicle type is '+db_vehicle_type
    print("The vehicle is Fake.The color and vehicle is altered.The original color is",db_vehicle_color,'The original vehicle model is',db_vehicle_type)
else:
  vehicle_result='Fake vehicle-no data found on database'
  print("The vehicle is fake.No vehicle found with the given vehicle number")
# print(vehicle_number)
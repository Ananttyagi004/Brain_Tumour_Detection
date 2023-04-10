
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import keras
from PIL import Image,ImageOps
import numpy as np
from keras_preprocessing.image import load_img,img_to_array
import cv2
import numpy as np
import os 
import h5py
import matplotlib.pyplot as plt

st.header("Brain Tumour Prediction")

def main():
   file= st.file_uploader('Choose the file',type=['jpg','png','jpeg'])
   if file is not None:
      image=Image.open(file)
      figure=plt.figure(figsize=(4,4))
      plt.imshow(image)
      plt.axis('off')
      result=predict_class(image)
      st.write(result)
      st.pyplot(figure)

def predict_class(image):
   model=keras.models.load_model('bestmodel.h5')
   shape=((224,224,3))
   
   test_image=image.resize((224,224))
   test_image=img_to_array(test_image)
   test_image=test_image/255.0
   test_image=np.expand_dims(test_image,axis=0)
   class_names=['Brain Tumour','Healthy']
   predictions=model.predict(test_image)[0][0]
   if predictions>=0.5:
      predictions='Healthy'

   else:
      predictions='Brain Tumour'  

   return predictions

if __name__=="__main__":
   main()
   
    
   


import streamlit as st
# import tensorflow as tf
# import tensorflow_hub as hub
import keras
from keras import models
import keras_preprocessing
from keras_preprocessing.image import img_to_array
from PIL import Image,ImageOps
import numpy as np
from keras.engine.functional import Functional
import tensorflow.compat.v2 as tf
# from keras_preprocessing.image import load_img,img_to_array
import numpy as np


st.header("Brain Tumour Prediction")

def main():
   file= st.file_uploader('Choose the file',type=['jpg','png','jpeg'])
   if file is not None:
      image=Image.open(file)
      result=predict_class(image)
      st.write(result)
      st.image (image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
      
      

def predict_class(image):
   model=models.load_model('bestmodel.h5')
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

# if __name__=="__main__":
#    main()
   
    
   


import streamlit as st
import io
import numpy as np
from PIL import Image
import tensorflow as tf
import efficientnet.tf.keras as efn
st.title('plant leaf disease classification')
st.write('based on image uploaded we will tell it is healthy or not')
gpus=tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0],True)
#some times cant load cnn type of errors may arise that can be solved by above 2 lines of code
model=tf.keras.models.load_model(r"C:\Users\PVSKSIDDHARDHA\model.h5")

#uploading file
upload_file=st.file_uploader('choose file from device',type=['png','jpg'])
predictions_map={0:'healthy_one',1:'multiple diseases',2:'has rust',3:'has scab'}
if upload_file is not None:
   image=Image.open(io.BytesIO(upload_file.read()))
#this prints image which we uploaded
   st.image(image,use_column_width=True)
   resized_image=np.array(image.resize((512,512)))/255.0
   #adding batch_dimension
   images_batch=resized_image[np.newaxis,:,:,:]
   predictions_array=model.predict(images_batch)
   predictions=np.argmax(predictions_array)
   results=f"The plant has {predictions_map[predictions]} with probability of {int(predictions_array[0][predictions]*100)}%"
   if predictions==0:
      st.success(results)
   else:
      st.error(results)
#this makes image size in allignment with above lines and image wont be that big
#above process convert image into io bytes for reading and preprocessing etc it will be useful
#but have to convert to numpy array for ml operations 


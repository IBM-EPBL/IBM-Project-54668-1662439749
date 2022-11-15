import os
import cv2
import numpy as np
import tensorflow as tf

graph= tf.compat.v1.get_default_graph()

class Video(object):
	def __init__(self):#This part of code will run while opening the app
		self.video = cv2.VideoCapture(0)
		self.roi_start = (50, 150)
		self.roi_end = (250, 350)
		self.model = tf.keras.models.load_model('asl_model.h5') # to Execute Local Trained Model
		# self.model = load_model('IBM_Communication_Model.h5') # to Execute IBM Trained Model
		self.index=['A','B','C','D','E','F','G','H','I']
		self.y=0

	def __del__(self):#This part of code will run while closing the app
		self.video.release()
		os.remove("region of interest image.jpg")

	def get_frame(self):#extracts frames from video
		ret,frame = self.video.read()
		frame = cv2.resize(frame, (640, 480))
		copy = frame.copy()
		copy = copy[150:150+200,50:50+200]
		# Prediction Start
		cv2.imwrite('region of interest image.jpg',copy)
		copy_img = tf.keras.preprocessing.image.load_img('region of interest image.jpg', target_size=(64,64))
		x = tf.keras.preprocessing.image.img_to_array(copy_img)
		x = np.expand_dims(x, axis=0)
		pred = np.argmax(self.model.predict(x), axis=1)
		self.y = pred[0]
		cv2.putText(frame,'The Predicted Alphabet is: '+str(self.index[self.y]),(100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
		ret,jpg = cv2.imencode('.jpg', frame)
		return jpg.tobytes()

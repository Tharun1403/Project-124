# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf
module = tf.keras.modules.load_module("keras_module.h5")


# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
		
		# expand the dimensions
		img = cv2.resize(frame,(224,224))
 		test_image = np.array(img,dtype=np.float32)
    	test_image = np.expand_dims(test_image,axis=0)
    	normalize_image=test_image/255
    	prediction=module.predict(normalize_image)
    	print(prediction)
		rock=int()
		paper=int()
		scissor=int()

		print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")


		# normalize it before feeding to the model
		
		# get predictions from the model
		
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()

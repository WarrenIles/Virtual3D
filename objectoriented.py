# This will be an object oriented version
# of the virtual 3d game.
import cv2
import numpy as np


class FaceFinder:
	""" Will utilize haar cascade filter to detect largest face from a frame """


	def __init__(self):
		print('FaceFinder initialize')
		self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


	def find_face(self, frame):
		""" Returns face center (x,y), draws rect on frame """
		# Convert into grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Detect faces
		#   detectMultScale returns an 2d ndarray
		faces = self.face_cascade.detectMultiScale(gray, minNeighbors = 9)
		print('detected face(s) at:', faces)

		# Draw rectangle around the faces
		if faces is None:
			return None
			
		bx=by=bw=bh = 0

		for (x, y, w, h) in faces:
			if w > bw:  # is current face bigger than biggest so far
				bx,by,bw,bh = x,y,w,h
		cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 255), 3)
		return (bx+bw/2), (by+bh/2)
  		

#--------------------------------------------
#main
#
# not sure if this is in right spot print('starting oo virtual3d')

ff = FaceFinder()
cap = cv2.VideoCapture(cv2.CAP_ANY)
if not cap.isOpened():
	print("Couldn't open cam")
	exit()





while True:
	retval, frame = cap.read()
	if retval == False:
		print("camera error!")

	ff.find_face(frame)
	cv2.imshow('q to quit',frame)

	if cv2.waitKey(30) == ord('q'):
		break


pause = input('press enter to end')
#destroy cam
cap.release()
cv2.destroyAllWindows()
#print('virtual3d complete')
print('starting oo virtual3D')
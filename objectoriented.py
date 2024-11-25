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
		faces = self.face_cascade.detectMultiScale(gray, MinNeighbors = 9)
		print('detected face(s) at:', faces)

		# Draw rectangle around the faces
		if faces is None:
			return None
			
		bx=by=bw=bh = 0

		for (x, y, w, h) in faces:
			if w > bw:  # is current face bigger than biggest so far
				bx,by,bw,bh = x,y,w,h
  		cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 255), 3)
  		return ((bx+bw//2), by+bh//2))
  		

#--------------------------------------------
#main
#
print('starting oo virtual3d')

ff = FaceFinder()
print('virtual3d complete')
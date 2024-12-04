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
		return (bx+bw//2), (by+bh//2)


class Stage:
    """Initialized with display size, draws background grid based on position"""
    def __init__(self):
     self.disp_h = 0
     self.disp_w = 0
     self.cam_h = 720
     self.cam_w = 1280
     self.save_x = 960

     def draw_target_xy(self, img, pos, size):
      cv2.circle(img, pos, size, (0,0,255), -1)
      cv2.circle(img,pos,int(size*.8), (255,255,255), -1)
      cv2.circle(img, pos,int(size*.6), (0,0,255), -1)
      cv2.circle(img, pos,int(size*.4), (255,255,255), -1)
      cv2.circle(img, pos,int(size*.2), (0,0,255), -1)


     def draw_targetz(self,pos,facexy):
     	tx,ty,tz = pos
     	cv2.circle(img, (ball0x, ball0y), 50, (255,0,0), -1)
     	cv2.line(img, (960+ int((600-960)*.3**2), 540 ),(ball0x, ball0y),(255,0,0),3)


     def update(self, facexy):
     	x,y = facexy
     	e = .9 # smoothing constant

     	x = e * x + (1-e)*self.save_x
     	self.save_x = save_x
     	img = np.zeros([1080,1920,3])
     	decay = .3
     	sx = sy = 0
     	dx = int ((x - self.cam_w/2)*2)
     	for i in range(1,7):
     		sx = sx + int((960-sx)*decay)
     		sy - sy + int((540-sy)*decay)
     		dx = int(dx * decay)
     		#print(sx, sy)
     		cv2.rectangle(img, (sx+dx,sy), (1920-sx+dx, 1080-sy),(255,255,255),1 )
  		

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
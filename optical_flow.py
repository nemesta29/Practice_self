#optical flow
#Use to avoid recounting of frames
import numpy as np
import cv2 as cv
import math

cap = cv.VideoCapture('test_video.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
   qualityLevel = 0.3,
   minDistance = 7,
   blockSize = 7 )
   
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
  maxLevel = 5,
  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
  
# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
#ret, old_frame = cap.read()
old_gray = cv.cvtColor(cv.imread('View3/image10.jpg',1),cv.COLOR_BGR2GRAY)
#old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
#p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_gray)

# top_left = [655, 640]#[600, 450]
# top_right = [655, 740]#[600, 710]
 
# bottom_right = [716, 506] #[710, 440]
# bottom_left =  [716, 640]#[710, 720]

# p0 = [top_left, top_right, bottom_right, bottom_left]

#For image convention
pt1 = [620, 718]
pt2 = [540, 718]
pt3 = [580, 718]
pt4 = [700, 718]
pt5 = [660, 718]

# #For frame convention
# pt1 = [635, 556]
# pt2 = [595, 556]
# pt3 = [555, 556]
# pt4 = [675, 556]
# pt5 = [715, 556]
p0 = [pt1, pt2, pt3, pt4, pt5] 

p0 = np.array(p0).reshape(-1,1,2).astype(np.float32)
p0_base = p0 
#print(p0)
count = 1

def findMagAndAngle(p0,p1):
	old_y, old_x = p0[...,0], p0[...,1]
	new_y, new_x = p1[...,0], p1[...,1]
	diff_y = np.subtract(new_y, old_y)
	diff_x = np.subtract(new_x, old_x)
	ang = np.arctan2(diff_y,diff_x) * 180 / np.pi
	mag = np.sqrt(diff_x*diff_x + diff_y*diff_y)
	return ang, mag
 
ctr = 0
while(count < 250):
	count += 1
	frameName = 'View3/image' + str(count)+'.jpg'
	frame = cv.imread(frameName,1)
	frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	# calculate optical flow
	p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
	# Select good points
	good_new = p1[st==1]
	good_old = p0[st==1]
	# draw the tracks
	for i,(new,old) in enumerate(zip(good_new,good_old)):
		a,b = new.ravel()
		c,d = old.ravel()
	#mask = cv.line(mask, (a,b),(c,d), (0,255,0), 2)
		frame = cv.circle(frame,(a,b),5,(0,255,0),-1)
	#img = cv.add(frame,mask)
	#draw_flow(frame_gray, good_new, good_old, step=16)
	cv.imshow('frame',frame)
	k = cv.waitKey(30) & 0xff
	if k == 27:
		break
	# Now update the previous frame and previous points
	old_gray = frame_gray.copy()
	p0 = good_new.reshape(-1,1,2)
	if p0.shape[0] >=2:
		for i in range(0, p0.shape[0]):
			# if math.fabs(int(p0[i][0][1])) <= 15:
			if math.fabs(int(p0[i][0][1])) - 230 <= 5:
				ctr+=1		
		if ctr>=2:
			p0 = p0_base
			ctr = 0
	else:
		print ('Error!\nInvalid trackers!')
		print (p0)
		print (p0.shape)
		break
cv.destroyAllWindows()
cap.release()

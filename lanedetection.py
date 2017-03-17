#Vijay D, SDC ND @ Udacity, Feb 2017 Cohort

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
#%matplotlib inline
debug = 0

print ('\n\nProject: P1, Vijay D. \nBasic Lane Detection')
print ('\nPICK A VIDEO:    -OR-')
print ('___________________________________________')
print ('1:  Video w/solid white right lane marker')
print ('2:  Video w/solid yellow left lane marker')
print ('3:  Video - Challenge (with 2 shadow zones)')
print ('\n\nPICK AN IMAGE:')
print ('___________________________________________')
print ('4:  White right lane marker')
print ('5:  Yellow left lane marker')
print ('6:  White curve marker')
print ('7:  Yellow left curve lane marker')
print ('8:  Yellow left curve lane marker 2')
print ('9:  Challenge image with shadow')
print ('10: White car lane switch')
choice = input('\nYour Choice (Enter:exit): ')


video = 0  
font = cv2.FONT_HERSHEY_SIMPLEX


# Assign an img or a video filename 
if   (choice == '1'): 	
    name = 'test_videos\solidwhiteright.mp4'
    video = 1
elif (choice == '2'):  
    name = 'test_videos\solidyellowleft.mp4'
    video = 1
elif (choice == '3'):  
    name = 'test_videos\challenge.mp4'
    video = 1
elif (choice == '4'): 	name   = 'test_images\solidWhiteRight.jpg'
elif (choice == '5'):   name   = 'test_images\solidYellowLeft.jpg'
elif (choice == '6'):   name   = 'test_images\solidWhiteCurve.jpg'
elif (choice == '7'):   name   = 'test_images\solidYellowCurve.jpg'
elif (choice == '8'):   name   = 'test_images\solidYellowCurve2.jpg'
elif (choice == '9'):   name   = 'test_images\shadow.jpg'
elif (choice == '10'):  name   = 'test_images\whiteCarLaneSwitch.jpg'
else:                  
    exit()

#Load video file
if (video):
    cap = cv2.VideoCapture(name)
    framenum = 1
    
else:  
    image = mpimg.imread(name) 
    
   
while True:
    if (video):
        ret, image  = cap.read()
        if ret == False: break
        if (debug): print ("Frame: ", framenum)
    
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
   
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 9 #13  #must be odd
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    if (debug):  cv2.imshow('Blur_Gray', blur_gray)

    mask_white = cv2.inRange(blur_gray, 150, 255)
    #use inRange() to perform color detection (by specifying the lower limit and upper limits of thecolor to detect)
        
    # Define parameters for Canny Edge Det
    low_threshold = 60 #60 old 50
    high_threshold = 180 #180 old 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold, apertureSize=3) #old = mask_white
    if (debug): cv2.imshow('Cannyedges', edges)
    
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges) 
    ignore_mask_color = 200   
    imshape = image.shape
    edge_offset = 100 #to move and retract from the edges
    center_offset = 10 #to locate the top trapezium points to the left/rt of (center, center) - the top vertices of the ROI

    #Left btm, Left top, Right top, Right btm: adding 35px to 1/2 the ht. from top
    vertices = np.array([[(edge_offset,imshape[0]), \
						  (imshape[1]/2 - center_offset, 35+imshape[0]/2), \
						  (imshape[1]/2 + center_offset, 35+imshape[0]/2), \
						  (imshape[1] - edge_offset,imshape[0])]], \
						  dtype=np.int32)
    if (debug): print ('Polygon vertices:', vertices[0][0],vertices[0][1],vertices[0][2],vertices[0][3] )

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
   
    # Define the Prob. Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2             # distance resolution in pixels of the Hough grid
    theta = np.pi/180   # angular resolution in radians of the Hough grid
    threshold = 20      # 15 minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100 # minimum number of pixels making up a line
    max_line_gap = 100    # 20 maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    
    #Two arrays for holding pos and neg slopes and their end points
    posslope = np.zeros((200, 5))
    negslope = np.zeros((200, 5))
    
    #Temp indices to count lines on the left, right, bad, total
    i = 0
    j = 0
    k = 0
    l = 0
   
    for line in lines:
      for x1,y1,x2,y2 in line:
        temp = (y2-y1)/(x2-x1)
        temp = round(temp,1)
        if (temp < -0.35): #LEFT LANE
            negslope[i][0]=temp
            negslope[i][1]=x1
            negslope[i][2]=y1
            negslope[i][3]=x2
            negslope[i][4]=y2
            negc = (y1 - temp*x1) #Calc intercept, c, for lines with neg slope, lazy man's way
            xnegcbtm = int((imshape[0]-negc)/temp)
            xnegctop = int((2*imshape[0]/3-negc)/temp)
            i += 1
        elif (temp > 0.35): #RIGHT LANE
            posslope[j][0]=temp
            posslope[j][1]=x1
            posslope[j][2]=y1
            posslope[j][3]=x2
            posslope[j][4]=y2
            posc = (y1 - temp*x1)#Calc intercept, c, for lines with neg slope
            xposcbtm = int((imshape[0]-posc)/temp )
            xposctop = int((2*imshape[0]/3-posc)/temp)
            j += 1
        else: #remove the line data of the line which does not belong here due to a strange slope
            np.delete(lines, l, 0)#this is only a msg. op is actually irrelevant becoz mod. array isn't being captured
            if (debug): print ('Bad line detected: outlier slope=', temp)
		
        if (abs(temp) > 0.35): #temporary statement that helps filter out outliers
             if (debug): 
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,255),10)
                cv2.circle(line_image, (x1,y1), 4, (0,255,0), -1)
                cv2.circle(line_image, (x2,y2), 4, (255,0,0), -1) 
             k += 1 #inrement the # of good lines
      l +=1 	
    
	
    if (debug):
        #print ('Lines w/negslope', negslope)#, '\nLines with positive slope', posslope)  
        print ('Tot. Lines:', l, '| L:', i, '| R:', j, '| Bad lines detected:', l-k)
        print ("Avg. Slope: L:", np.sum(negslope[:,0])/i, '| R:', np.sum(posslope[:,0])/j)
        print ('Left lane Btm X pt.(', xnegcbtm, ') Top X pt. (', xnegctop, ')')
        print ('Right lane Btm X pt.(', xposcbtm, ') Top X pt. (', xposctop, ')')
        cv2.circle(line_image, (x1,y1), 10, (255,255,0), -1)
        cv2.circle(line_image, (x2,y2), 10, (255,255,0), -1)
    #Drawing the two lane lines
    cv2.line(line_image,(xnegcbtm,int(imshape[0])),(xnegctop,int(2*imshape[0]/3)),(255,0,255),25)
    cv2.line(line_image,(xposcbtm,int(imshape[0])),(xposctop,int(2*imshape[0]/3)),(255,0,255),25)
    
    # Create a "color" binary image to combine with line image : DOES NOT DO ANYTHING, same as CANNY_EDGES, ASK Q
    #color_edges = np.dstack((edges, edges, edges)) 
    #cv2.imshow('color_edges', color_edges)

    # Draw the lines on the edge image; use this for overlaying an image on another with alpha blending, aka transparency
    if (video): 
        fram = 'Frame #:'+ str(framenum)
    totlines = 'Total detected lines:'+str(l)+' Actual mapped:'+str(k)+' (Left:'+str(i)+' Right:'+str(j)+')'
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    quitinstr ='Press Ctrl+C to quit processing'

    if (video): 
        cv2.putText(lines_edges, fram, (10,15), font, 0.5, (255,255,255), 0, cv2.LINE_AA)
        
    cv2.putText(lines_edges, totlines, (10,30), font, 0.5, (0,255,255), 0, cv2.LINE_AA)
    cv2.putText(lines_edges, quitinstr, (imshape[1]-270,30), font, 0.5, (0,0,255), 0, cv2.LINE_AA)
    cv2.imshow('lines_edges', lines_edges)
    
    if (video): 
        framenum+=1
    if (video != 1):
        break
    else:
        kk = cv2.waitKey(5) & 0xFF
        if kk == 27: #ESC key
            break

if (debug == 0): print ('Debug flag can be turned ON (=1) inside code for interim output')

if (video): cap.release()
else: 
    
    kk = cv2.waitKey()
    
    
cv2.destroyAllWindows()
import cv2
import sys
import pdb
import pickle
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
from scipy.spatial import distance as dist
from scipy import optimize
import math
from scipy.interpolate import interp1d
from numpy.linalg import norm
import matplotlib.pyplot as plt
import os


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


class openCVImageProcessingTools:
    def __init__(self,analysisLoc, figureLoc, ff, showI = False):
        #  "/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/170606_f37/170606_f37_2017.07.12_001_behavingMLI_000_raw_behavior.avi"
        self.analysisLocation = analysisLoc
        self.figureLocation = figureLoc
        self.f = ff
        self.showImages = showI
        self.Vwidth = 816
        self.Vheight = 616

    ############################################################
    def __del__(self):
        self.video.release()

        cv2.destroyAllWindows()
        print('on exit')


    ############################################################
    def openVideo(self,pathAndFileName,outputProps=True):

        # get properties
        self.video = cv2.VideoCapture(pathAndFileName)
        self.Vlength = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.Vwidth = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Vheight = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.Vfps = self.video.get(cv2.CAP_PROP_FPS)
        if outputProps:
            print('Video properties : %s frames, %s pixels width, %s pixels height, %s fps' % (self.Vlength, self.Vwidth, self.Vheight, self.Vfps))

        if not self.video.isOpened():
            print('Could not open video')
            sys.exit()
        # read first video frame
        ok, img = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        # Return an array representing the indices of a grid.
        self.imgGrid = np.indices((self.Vheight, self.Vwidth))

        return (img)
    ############################################################
    def findBarsCrossCorr(self, firstImgGray, positions):
        polyCoeffsBelow = np.array([  2.48892127e-08,   6.18721000e-05,  -5.37723176e-02,9.26067051e+01])
        polyCoeffsAbove = np.array([ -1.66661173e-08,   9.26715756e-05,  -5.73128529e-02,8.80751777e+01])
        barWidth = 15




    ############################################################
    def findHorizontalArea(self, img,Ycoordinates=None):

        if Ycoordinates is None:
            yPos = 300
            barWidth = 10
        else:
            yPos = Ycoordinates[0]
            barWidth = Ycoordinates[1]

        Npix = 5
        continueLoop = True
        # optimize with keyboard
        while continueLoop:
            #rungs = []
            imgLine = img.copy()
            cv2.line(imgLine, (0, yPos), (self.Vwidth, yPos), (255, 0, 255), 2)
            cv2.line(imgLine, (0, yPos+barWidth), (self.Vwidth, yPos+barWidth), (255, 0, 255), 2)

            cv2.imshow("Rungs", imgLine)
            #print 'current xPosition, yPostion : ', xPosition, yPosition
            PressedKey = cv2.waitKey(0)
            if PressedKey == 56 or PressedKey ==82: #UP arrow
                yPos -= Npix
            elif PressedKey == 50 or PressedKey ==84: #DOWN arrow
                yPos += Npix
            elif PressedKey == 54 or PressedKey ==83: #RIGHT arrow
                barWidth += Npix
            elif PressedKey == 52 or PressedKey ==81: #LEFT arrow
                barWidth -= Npix
            elif PressedKey == 13 or PressedKey == 32: # Enter or Space
                continueLoop = False
            elif PressedKey == 27: # Escape
                continueLoop = False
            else:
                pass
            cv2.destroyWindow("Rungs")


        print('Pos, width Y :', yPos,barWidth)
        #mask = np.zeros((self.Vheight, self.Vwidth))
        return (yPos,barWidth)


    ############################################################
    def cropImg(self, img,Ycoordinates=None):

        if Ycoordinates is None:
            leftY = 215
            rightY = 215
            Npix = 5
            continueLoop = True
            # optimize with keyboard
            while continueLoop:
                #rungs = []
                imgLine = img.copy()
                cv2.line(imgLine, (0, leftY), (self.Vwidth, rightY), (255, 0, 0), 2)

                cv2.imshow("Rungs", imgLine)
                #print 'current xPosition, yPostion : ', xPosition, yPosition
                PressedKey = cv2.waitKey(0)
                if PressedKey == 56 or PressedKey ==82: #UP arrow
                    leftY -= Npix
                elif PressedKey == 50 or PressedKey ==84: #DOWN arrow
                    leftY += Npix
                elif PressedKey == 54 or PressedKey ==83: #RIGHT arrow
                    rightY += Npix
                elif PressedKey == 52 or PressedKey ==81: #LEFT arrow
                    rightY -= Npix
                elif PressedKey == 13 or PressedKey == 32: # Enter or Space
                    continueLoop = False
                elif PressedKey == 27: # Escape
                    continueLoop = False
                else:
                    pass
                cv2.destroyWindow("Rungs")
        else:
            leftY = Ycoordinates[0]
            rightY = Ycoordinates[1]

        print('Left, right Y :', leftY,rightY)
        mask = np.zeros((self.Vheight, self.Vwidth))

        # create masks for mouse area and for lower area
        maskGrid = ((self.imgGrid[1] - 0.)*(rightY-leftY) - (self.imgGrid[0] -leftY)*(self.Vwidth - 0)) < 0
        #maskInv = np.sqrt((imgGrid[1] - xCenter) ** 2 + (imgGrid[0] - yCenter) ** 2) < Radius
        mask[maskGrid] = 1
        #wheelMaskInv[maskInv] = 1
        mask = np.array(mask, dtype=np.uint8)
        #wheelMaskInv = np.array(wheelMaskInv, dtype=np.uint8)
        return (mask)


    ############################################################
    def trackRungs(self, mouse, date, rec, **kwargs):
        badVideo = 0
        stopProgram = False
        # tracking parameters #########################
        self.thresholdValue = 0.7  # in %
        self.minContourArea = 40  # square pixels

        self.scoreWeights = {'distanceWeight': 5, 'areaWeight': 1}
        ###############################################
        rec = rec.replace('/', '-')
        videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior.avi' % (mouse, date, rec)
        firstImg = self.openVideo(videoFileName)
        firstImgGray = cv2.cvtColor(firstImg, cv2.COLOR_BGR2GRAY)
        print('image dims :', np.shape(firstImgGray))
        # create video output streams
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.outRung = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_rungTracking.avi' % (mouse, date, rec), fourcc, 20.0, (self.Vwidth, self.Vheight))


        # Crop image ######################
        #mask = self.cropImg(firstImg,[215,215])
        #imgBottom = cv2.bitwise_and(firstImg, firstImg, mask=mask)
        #cv2.imshow("Crop", imgBottom)
        #PressedKey = cv2.waitKey(0)
        #cv2.destroyWindow("Crop")

        # Find horizontal areas for paw position extraction ###
        (yPosAbove,barWidthAbove) = self.findHorizontalArea(firstImgGray,[225,20])
        (yPosBelow,barWidthBelow) = self.findHorizontalArea(firstImgGray,[565,20])

        self.findBarsCrossCorr(firstImgGray,[yPosAbove,barWidthAbove,yPosBelow,barWidthBelow])

        plt.plot(np.average(firstImgGray[yPosAbove:(yPosAbove+barWidthAbove)],0),label='above')
        plt.plot(np.average(firstImgGray[yPosBelow:(yPosBelow+barWidthBelow)],0),label='below')

        plt.legend()
        plt.show()
        pdb.set_trace()
        # Save some images #################
        saveImgIdx = [0,1000,4000]
        idx = 0
        while True:
            ok, img = self.video.read()
            if not ok:
                break

            if idx in saveImgIdx:

                #plt.plot(average(img[])
                cv2.imwrite( self.analysisLocation + '%s_%s_%s_image#%s.png' %(mouse,date,rec,idx), img )
            idx+=1

        pdb.set_trace()
        # Find lines in image #############
        imgRungs = firstImg.copy()
        imgBottomGray = cv2.cvtColor(imgBottom, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(imgBottomGray,50,150 ,apertureSize = 3)
        minLineLength = 40
        maxLineGap = 20
        cv2.imshow('Edges', edges)
        PressedKey = cv2.waitKey(0)
        cv2.destroyWindow('Edges')
        #pdb.set_trace()
        lines = cv2.HoughLinesP(edges,1,np.pi/(2*180),15,minLineLength,maxLineGap)
        if lines is not None:
            #pdb.set_trace()
            print('number of detected lines :',len(lines[0]))
            #for rho,theta in lines[0]:
            for x1, y1, x2, y2 in lines[0]:
                # a = np.cos(theta)
                # b = np.sin(theta)
                # x0 = a*rho
                # y0 = b*rho
                # x1 = int(x0 + 1000*(-b))
                # y1 = int(y0 + 1000*(a))
                # x2 = int(x0 - 1000*(-b))
                # y2 = int(y0 - 1000*(a))
                cv2.line(imgRungs,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imshow('Rungs', imgRungs)
        PressedKey = cv2.waitKey(0)
        cv2.destroyWindow('Rungs')





    ############################################################
    def trackPawsAndRungs(self,mouse,date,rec, **kwargs):
        badVideo = 0
        stopProgram = False
        # tracking parameters #########################
        self.thresholdValue = 0.7 # in %
        self.minContourArea = 40 # square pixels

        self.scoreWeights = {'distanceWeight':5,'areaWeight':1}
        ###############################################
        rec = rec.replace('/', '-')
        videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior.avi' % (mouse, date, rec)
        firstImg = self.openVideo(videoFileName)

        # create video output streams
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.outPaw = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_pawTracking.avi' % (mouse, date, rec), fourcc, 20.0, (self.Vwidth, self.Vheight))
        self.outRung  = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_rungTracking.avi' % (mouse, date, rec),fourcc, 20.0, (self.Vwidth, self.Vheight))

        # read first video frame
        ok, img = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        
        # Return an array representing the indices of a grid.
        imgGrid = np.indices((self.Vheight, self.Vwidth))

        ########################################################################
        # loop to find correct wheel mask
        Radius = 1500  # 1400
        xCenter = 1205  # 1485
        yCenter = 1625  # 1545
        xPosition = 190
        yPosition = 0
        Npix = 5
        nIt = 0
        ConfirmedMask=False

        if 'WheelMask' in kwargs:
            Radius = kwargs['WheelMask'][0]
            xCenter = kwargs['WheelMask'][1]
            yCenter = kwargs['WheelMask'][2]

        
        while not ConfirmedMask and not stopProgram:

            imgCircle = img.copy()
            cv2.circle(imgCircle, (xCenter, yCenter), Radius, (0, 0, 255), 2)
            #if nIt > 0:
                #cv2.circle(imgCircle, (oldxCenter, oldyCenter), oldRadius, (0, 0, 100), 2)
                #cv2.putText(imgCircle,'now',(10,10),color=(0, 0, 255))
                #cv2.putText(imgCircle,'before',(10,20),fontScale=4,color=(0, 0, 150),thickness=2)
            cv2.imshow("Wheel mask", imgCircle)
            #print 'current radius, xCenter, yCenter : ' , Radius, xCenter, yCenter
            #print('Adjust the wheel mask using the arrows and +/- \n Press Space or Enter to confirm')
            PressedKey = cv2.waitKey(0)
            if PressedKey == 56 or PressedKey ==82: #UP arrow
                yCenter -= Npix
            elif PressedKey == 50 or PressedKey ==84: #DOWN arrow
                yCenter += Npix
            elif PressedKey == 54 or PressedKey ==83: #RIGHT arrow
                xCenter += Npix
            elif PressedKey == 52 or PressedKey ==81: #LEFT arrow
                xCenter -= Npix
            elif PressedKey == 43: # + Button
                Radius += Npix
            elif PressedKey == 45: # - Button
                Radius -= Npix
            elif PressedKey == 13 or PressedKey == 32: # Enter or Space
                cv2.destroyWindow("Wheel mask")
                ConfirmedMask = True
            elif PressedKey == 27: # Escape
                cv2.destroyWindow("Wheel mask")
                stopProgram=True
                #sys.exit()
            elif PressedKey ==8: #Backspace marks the recording as bad
                badVideo = 1
                cv2.destroyWindow("Wheel mask")
                #cv2.destroyAllWindows()
                return stopProgram, [Radius, xCenter, yCenter], [xPosition,yPosition], badVideo
            else:
                pass
            #nIt +=1
        print('masking after loop, Radius = %s, xCenter %s, yCenter = %s' % (Radius, xCenter, yCenter))

        wheelMask = np.zeros((self.Vheight, self.Vwidth))
        wheelMaskInv = np.zeros((self.Vheight, self.Vwidth))

        # create masks for mouse area and for lower area
        mask = np.sqrt((imgGrid[1] - xCenter) ** 2 + (imgGrid[0] - yCenter) ** 2) > Radius
        maskInv = np.sqrt((imgGrid[1] - xCenter) ** 2 + (imgGrid[0] - yCenter) ** 2) < Radius
        wheelMask[mask] = 1
        wheelMaskInv[maskInv] = 1
        wheelMask = np.array(wheelMask, dtype=np.uint8)
        wheelMaskInv = np.array(wheelMaskInv, dtype=np.uint8)

        ########################################################################
        xPosition = 190
        yPosition = 0

        if 'RungsLoc' in kwargs:
            xPosition = kwargs['RungsLoc'][0]
            yPosition = kwargs['RungsLoc'][1]
            


        # loop to find correct rung lines
        imgInv = cv2.bitwise_and(img, img, mask=wheelMaskInv)
        # convert image to gray-scale
        imgGWheel = cv2.cvtColor(imgInv, cv2.COLOR_BGR2GRAY)
        nIt = 0
        ConfirmedRungALignement = False
        while not ConfirmedRungALignement and not stopProgram:
            rungs = []
            imgLines = imgCircle.copy()
            circles = cv2.HoughCircles(imgGWheel, cv2.HOUGH_GRADIENT, 1, minDist=150, param1=50, param2=15, minRadius=30, maxRadius=40)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                cLoc = []
                for i in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(imgLines, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv2.circle(imgLines, (i[0], i[1]), 2, (0, 0, 255), 3)
                    # cv2.circle(orig3, (i[0], i[1]), 2, (0, 0, 255), 3)
                    cv2.line(imgLines, (i[0], i[1]), (xPosition,yPosition), (255, 0, 0), 2)
                    rungs.append([0, i[0], i[1], xPosition, yPosition])
                    if nIt > 0:
                        cv2.line(imgLines, (i[0], i[1]), (oldxPosition, oldyPosition), (100, 0, 0), 2)
                    cLoc.append([i[0],i[1]])
            cLoc =np.asarray(cLoc)
            a = np.sum(np.sqrt((cLoc[0]-cLoc[1])**2)) #np.linalg.norm(cLoc[0]-cLoc[1])
            b = np.sum(np.sqrt((cLoc[0]-cLoc[2])**2))
            c = np.sum(np.sqrt((cLoc[1]-cLoc[2])**2))
            #pdb.set_trace()
            #print 'argLengths : ', a ,b, c
            cv2.imshow("Rungs", imgLines)
            #print 'current xPosition, yPostion : ', xPosition, yPosition
            PressedKey = cv2.waitKey(0)
            if PressedKey == 56 or PressedKey ==82: #UP arrow
                yPosition -= Npix
            elif PressedKey == 50 or PressedKey ==84: #DOWN arrow
                yPosition += Npix
            elif PressedKey == 54 or PressedKey ==83: #RIGHT arrow
                xPosition += Npix
            elif PressedKey == 52 or PressedKey ==81: #LEFT arrow
                xPosition -= Npix
            elif PressedKey == 13 or PressedKey == 32: # Enter or Space
                cv2.destroyWindow("Rungs")
                ConfirmedRungALignement = True
            elif PressedKey == 27: # Escape
                cv2.destroyWindow("Rungs")
                stopProgram=True
                #sys.exit()
            elif PressedKey ==8: #Backspace marks the recording as bad
                badVideo = 1
                cv2.destroyWindow("Wheel mask")
                return stopProgram, [Radius, xCenter, yCenter], [xPosition,yPosition], badVideo
            else:
                pass
        #########################################################################

        recs = []
        if not stopProgram:
            bboxFront = cv2.selectROI("Select dot for FRONT paw \n mouse %s - rec = %s // %s" % (mouse, date, rec), img, False)
            print(recs)
            bboxHind = cv2.selectROI("Select dot for HIND paw", img, False)
            cv2.destroyAllWindows()
            #print 'front, hind paw bounding boxes : ', bboxFront, bboxHind
            pointLoc = bboxFront[:2]
            #print 'bounding box area : ', bboxFront[2] * bboxFront[3]
            frontpawPos = []
            hindpawPos = []
            # append first paw postions to list : [0 number of image, 1 success or failure, 2 location of paw, 3 all roi - ellipse - info, 4 area ]
            frontpawPos.append([0, 's', bboxFront[:2], [], np.pi * bboxFront[2] * bboxFront[3] / 4.])
            hindpawPos.append([0, 's', bboxHind[:2], [], np.pi * bboxHind[2] * bboxHind[3] / 4.])
            #hindPawPos.append([0, [], bboxHind[:2], np.pi * bboxHind[2] * bboxHind[3] / 4.])
            fcheckPos = -1

        ########################################################################

        
        hcheckPos = -1
        nF = 1
        #########################################################################
        # loop over all images in video
        print("Mouse : %s\nRecording : %s // %s" % (mouse, date, rec))
        print("Running the tracking algorithm...")
        while not stopProgram:
            #os.system('clear')
            #print("Mouse : %s\nRecording : %s // %s" % (mouse, date, rec))
            # Read a new frame
            thresholdV = self.thresholdValue
            ok, img = self.video.read()
            if not ok:
                break
            orig = img.copy()
            origCL = img.copy()
            #orig3 = img.copy()
            # while (1):
            # ret, frame = cap.read()

            # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgInv = cv2.bitwise_and(img, img, mask=wheelMaskInv)
            imgMouse = cv2.bitwise_and(img, img, mask=wheelMask)

            # convert image to gray-scale
            imgGWheel = cv2.cvtColor(imgInv, cv2.COLOR_BGR2GRAY)
            imgGMouse = cv2.cvtColor(imgMouse, cv2.COLOR_BGR2GRAY)
            ###############################################################################################
            # find location of rungs


            # find circles in the lower part of the image, i.e., find screws to determine paw positions,
            circles = cv2.HoughCircles(imgGWheel, cv2.HOUGH_GRADIENT, 1, minDist=150, param1=50, param2=15, minRadius=30, maxRadius=40)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(origCL, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv2.circle(origCL, (i[0], i[1]), 2, (0, 0, 255), 3)
                    #cv2.circle(orig3, (i[0], i[1]), 2, (0, 0, 255), 3)
                    cv2.line(origCL, (i[0], i[1]), (xPosition, yPosition), (255, 0, 0), 3)
                    rungs.append([nF, i[0], i[1], xPosition, yPosition])
            # find lines in the upper part of the image, i.e., the rungs
            #edges = cv2.Canny(imgGMouse,10,150,apertureSize = 3)
            #minLineLength = 50
            #maxLineGap = 10
            #cv2.imshow('edges detection', edges)
            #pdb.set_trace()
            #lines = cv2.HoughLinesP(edges,1,np.pi/(2*180),50,minLineLength,maxLineGap)
            #if lines is not None:
            #    for x1,y1,x2,y2 in lines[0]:
            #        cv2.line(origCL,(x1,y1),(x2,y2),(0,255,0),2)

            self.outPawRung.write(origCL)
            if self.showImages:
                cv2.imshow("detected circles  - mouse : %s   rec : %s/%s" % (mouse, date, rec), origCL)

            #################################################################################################
            # find contours based on maximal illumination

            # blur image and apply threshold
            blur = cv2.GaussianBlur(imgGMouse, (5, 5), 0)
            minMaxL = cv2.minMaxLoc(blur)
            #mask = np.zeros(imgGMouse.shape,dtype="uint8")
            #cv2.drawContours(mask, [contour], -1, 255, -1)
            #mean,stddev = cv2.meanStdDev(image,mask=mask)
            while True:
                ret, th1 = cv2.threshold(blur, minMaxL[1] * thresholdV, 255, cv2.THRESH_BINARY)
                #print ret, th1
                # perform edge detection, then perform a dilation + erosion to
                # close gaps in between object edges
                edged = cv2.Canny(th1, 50, 100)
                edged = cv2.dilate(edged, None, iterations=1)
                edged = cv2.erode(edged, None, iterations=1)

                cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                if len(cnts)>0:
                    # sort the contours from left-to-right and initialize the
                    # 'pixels per metric' calibration variable
                    (cnts, _) = contours.sort_contours(cnts)
                nLarge = 0
                for c in cnts:
                    if cv2.contourArea(c) > self.minContourArea:
                        nLarge += 1
                if nLarge >= 4 :
                    break
                else:
                    thresholdV = thresholdV - 0.05
            #print mmean, sstddev
            #for c in cnts:
            #    mask = np.zeros(imgGMouse.shape, dtype="uint8")
            #    cv2.drawContours(mask, c, 0, 255, 2)
            #    mmean, sstddev = cv2.meanStdDev(imgGMouse, mask=mask)
            #    print cv2.contourArea(c), mmean, sstddev
            #pdb.set_trace()

            #cntDistances = []
            statsRois = []
            rois = []
            for c in cnts:
                # if the contour is not sufficiently large, ignore it
                if cv2.contourArea(c) < self.minContourArea:
                    continue

                # print 'contourArea : ',cv2.contourArea(c)
                # compute the rotated bounding box of the contour
                ell = cv2.fitEllipse(c)
                # print ell
                #cntDistances.append(dist.euclidean(pawPos[checkPos][3], ell[0]))
                #cntArea.append(np.pi * (ell[1][0] / 2.) * (ell[1][1] / 2.0))
                # get statistics of contour
                mask = np.zeros(imgGMouse.shape, dtype="uint8")
                cv2.drawContours(mask, c, -1, (255), 1)  # cv.drawContours(mask, contours, -1, (255),1)
                mmean, sstddev = cv2.meanStdDev(imgGMouse, mask=mask)
                statsRois.append([cv2.contourArea(c), mmean[0][0], sstddev[0][0]])

                rois.append([ell,np.pi * (ell[1][0] / 2.) * (ell[1][1] / 2.0)])
                orig = cv2.ellipse(orig, ell, (255, 0, 0), 2)


            #pdb.set_trace()

            # find ellipse which is the best continuation of the previous ones
            # print 'nContours, Dist, Areaa : ', nCnts, cntDistances, cntArea
            #print 'frame ', nF, len(rois),
            # if rois were detected
            if len(rois) > 0:
                cornerDist = []
                frontDist  = []
                hindDist   = []
                frontAreaChange = []
                hindAreaChange  = []
                fpScore = np.zeros(len(rois))
                hpScore = np.zeros(len(rois))
                #pdb.set_trace()
                for i in range(len(rois)):
                    cornerDist.append(dist.euclidean((0,self.Vheight), rois[i][0][0]))
                    frontDist.append(dist.euclidean(frontpawPos[fcheckPos][2], rois[i][0][0]))
                    hindDist.append(dist.euclidean(hindpawPos[hcheckPos][2], rois[i][0][0]))
                    frontAreaChange.append(np.abs(frontpawPos[fcheckPos][2] - statsRois[i][0]))
                    hindAreaChange.append(np.abs(hindpawPos[hcheckPos][2] - statsRois[i][0]))
                    #pdb.set_trace()
                    cv2.putText(orig, '%s , %s , %s' % (statsRois[i][0],statsRois[i][1],statsRois[i][2]), (int(rois[i][0][0][0]),int(rois[i][0][0][1])), cv2.QT_FONT_NORMAL, 0.4, color=(220, 220, 220))
                # score based on the distance to last paw position, smallest distance -> highest score
                fpScore+= self.scoreWeights['distanceWeight']/np.asarray(frontDist)
                hpScore+= self.scoreWeights['distanceWeight']/np.asarray(hindDist)
                # score based on change of contour area, smallest change -> highest score
                fpScore+= self.scoreWeights['areaWeight']/np.asarray(frontAreaChange)
                hpScore+= self.scoreWeights['areaWeight']/np.asarray(hindAreaChange)
                # score based on distance from lower left corner : small dist -> high score for hind
                #if len(cornerDist) == 2:
                #    hindIdx =  np.argmin(np.asarray(cornerDist))
                #    frontIdx = np.argmax(np.asarray(cornerDist))
                #else :

                highestSTD = np.argsort(1./(np.asarray(statsRois)[:,2]))
                hindSortIdx = np.argsort(np.asarray(hindDist))
                frontSortIdx = np.argsort(np.asarray(frontDist))
                cornerSortIdx = np.argsort(np.asarray(cornerDist))
                #frontIdx = highestSTD[0] # int(np.where(highestSTD == 0)[0])
                #hindIdx = highestSTD[1] # int(np.where(highestSTD == 1)[0])
                #pdb.set_trace()
                # if STD of first two points is very large : certain that they are the paws
                if statsRois[highestSTD[1]][2]/statsRois[highestSTD[2]][2] > 3. :
                    if np.where(highestSTD[0] == hindSortIdx) < np.where(highestSTD[0] == frontSortIdx):
                        hindIdx  = highestSTD[0]
                        frontIdx = highestSTD[1]
                    else:
                        hindIdx  = highestSTD[1]
                        frontIdx = highestSTD[0]
                elif np.where(hindSortIdx[0] == cornerSortIdx) < np.where(frontSortIdx[0] == cornerSortIdx):
                    hindIdx  = hindSortIdx[0]
                    frontIdx = frontSortIdx[0]
                else:
                    hindIdx  = 0
                    frontIdx = 0
                #if hindSortIdx[0] in highestSTD[:2] :
                #    hindIdx = hindSortIdx[0]
                if (frontDist[frontIdx] < abs(fcheckPos) * 50.): # frontIdx in np.where(highestSTD<=1)[0] :
                    #print 'front, hind index ', frontIdx, hindIdx
                    #pdb.set_trace()
                    # if (frontDist[frontIdx] < abs(fcheckPos) * 50.):
                    Str = 'frontDist success : %s' % frontDist[frontIdx]
                    frontpawPos.append([nF,rois[frontIdx][0][0],rois[frontIdx][0],rois[frontIdx][1],statsRois[frontIdx][0]])
                    fcheckPos = -1
                else:
                    Str = 'frontDist failure : %s' % frontDist[frontIdx]
                    frontpawPos.append([nF,rois[frontIdx][0][0],rois[frontIdx][0],rois[frontIdx][1],statsRois[frontIdx][0]])
                    fcheckPos -= 1
                #
                if (hindDist[hindIdx] < abs(hcheckPos) * 50.): # hindIdx in np.where(highestSTD<=1)[0] : #hindIdx in np.where(highestSTD<=1)[0] : #(hindDist[hindIdx] < abs(hcheckPos) * 50.):

                    Str2 = '    ///     hindDist success : %s' % hindDist[hindIdx]
                    hindpawPos.append([nF,rois[hindIdx][0][0], rois[hindIdx],rois[hindIdx][1],statsRois[hindIdx][0]] )
                    hcheckPos = -1
                else:
                    Str2 = 'hindDist failure : %s' % hindDist[hindIdx]
                    hindpawPos.append([nF, 'f', rois[hindIdx][0][0], rois[hindIdx],rois[hindIdx][1],statsRois[hindIdx][0]] )
                    hcheckPos -=1
                ##
                if self.showImages:
                    FrameStr =  'frame %s (len(rois) = %s)' % (nF, len(rois))
                    x =  int(self.Vwidth * (nF/float(self.Vlength)))
                    cv2.rectangle(orig, (0, self.Vheight), (x, self.Vheight-15), (100, 100, 100), thickness=-1)
                    cv2.putText(orig, FrameStr, (0, self.Vheight-20), cv2.QT_FONT_NORMAL, 0.45, color=(255, 255, 255))
                    cv2.putText(orig, Str, (0, self.Vheight-5), cv2.QT_FONT_NORMAL, 0.4, color=(220, 220, 220))
                    cv2.putText(orig, Str2, (int(self.Vwidth/3), self.Vheight-5), cv2.QT_FONT_NORMAL, 0.4, color=(220, 220, 220))
                    cv2.putText(orig,'frontpaw',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,color=(0, 255, 0))
                    orig = cv2.ellipse(orig, rois[frontIdx][0], (0, 255, 0), 2)
                    cv2.putText(orig,'hindpaw',(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,color=(0, 0, 255))
                    orig = cv2.ellipse(orig, rois[hindIdx][0], (0, 0, 255), 2)
                #print 'hind ',
                #dret = decideonAndAddPawPositions(hindpawPos, hcheckPos, rois)
                #hindpawPos.append(dret[1])
                #hcheckPos += dret[0]
                #if not dret[0]:
                #if self.showImages:
                #    cv2.putText(orig,'hindpaw',(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,color=(0, 0, 255))
                #    orig = cv2.ellipse(orig, rois[hindIdx][0], (0, 0, 255), 2)
                # pdb.set_trace()
                #print ' '
                #Dchange = abs(np.asarray(cntDistances))
                #Achange = abs(np.asarray(cntArea) / pawPos[checkPos][3] - 1.) * 100.  # in percent
                #DWeight = 0.5
                #print checkPos, Dchange, Achange,
                #Pindex = np.argmin(DWeight * Dchange + (1. - DWeight) * Achange)
                # Dindex = np.argmin(abs(np.asarray(cntDistances)-Dprojection))
                # print Dindex, Dprojection, pawPos[-2:]
                # if abs(cntArea[Dindex]/pawPos[-1][3] - 1.) < 0.2:
                # print cntDistances[Aindex]
                #frontpawPos.append([nF,rois[PindMax][0], rois[PindMax][1]])
                #hindpawPos.append([nF, rois[PindMin][0], rois[PindMin][1]])
                #if (Dchange[Pindex] < abs(checkPos) * 60.) and (Achange[Pindex] < abs(checkPos) * 200.):
                #    print 'success', Pindex, Dchange[Pindex], Achange[Pindex],
                #    return (0,[nF, rois[Pindex][0], rois[Pindex][0][0], cntArea[Pindex]])
                #cntFrontDistances = []
                #cntArea = []
                # pdb.set_trace()
                #for i in range(len(rois)):
                #    cntFrontDistances.append(dist.euclidean(pawPos[checkPos][2], rois[i][0][0]))
                #    cntFrontDistances.append(dist.euclidean(pawPos[checkPos][2], rois[i][0][0]))
                #    cntArea.append(rois[i][1])

                #Dchange = abs(np.asarray(cntDistances))
                #Achange = abs(np.asarray(cntArea) / pawPos[checkPos][3] - 1.) * 100.  # in percent
                #DWeight = 0.9
                # print checkPos, Dchange, Achange,
                #Pindex = np.argmin(DWeight * Dchange + (1. - DWeight) * Achange)
                # Dindex = np.argmin(abs(np.asarray(cntDistances)-Dprojection))
                # print Dindex, Dprojection, pawPos[-2:]
                # if abs(cntArea[Dindex]/pawPos[-1][3] - 1.) < 0.2:
                # print cntDistances[Aindex]
                #if (Dchange[Pindex] < abs(checkPos) * 60.) and (Achange[Pindex] < abs(checkPos) * 200.):
                #    print 'success', Pindex, Dchange[Pindex], Achange[Pindex],
                #    return (0, [nF, rois[Pindex][0], rois[Pindex][0][0], cntArea[Pindex]])
                #    # pawPos.append([nF, rois[Pindex], rois[Pindex][0], cntArea[Pindex]])

                #    # orig2 = cv2.ellipse(orig2, rois[Pindex], (0, 255, 0), 2)
                #    # orig3 = cv2.ellipse(orig3, rois[Pindex], (0, 255, 0), 2)
                #checkPos = -1  # pointLoc = rois[Dindex][0]  # maxStepCurrent = maxStep
                #else:
                #    print 'failure', Pindex, Dchange[Pindex], Achange[Pindex],
                #    return (1, [nF, -1, -1, -1])  # pawPos.append([nF, '', -1, -1, -1])  # checkPos -= 1


                #print 'front ',
                #dret = decideonAndAddPawPositions(frontpawPos,fcheckPos,rois)

            else:
                print('failure no rois')
                frontpawPos.append([nF, 'f', [-1,-1], -1, -1])
                hindpawPos.append([nF,'f', [-1,-1], -1, -1])
                fcheckPos -= 1
                hcheckPos -= 1
            # show image with all detected rois, and rois decided to be paws

                
            self.outPaw.write(orig)

            if self.showImages:
                cv2.imshow("Paw-tracking monitor - mouse : %s   rec : %s/%s" % (mouse, date, rec), orig)

            # wait and abort criterion, 'esc' allows to stop
            k = cv2.waitKey(1) & 0xff
            #print k
            if k == 27: break
            elif k ==8: #Backspace marks the recording as bad
                badVideo = 1
                cv2.destroyAllWindows()
                return stopProgram, [Radius, xCenter, yCenter], [xPosition,yPosition], badVideo
            elif k == 32:
                pdb.set_trace()
                #cv2.waitKey(100)
            nF += 1

        cv2.destroyAllWindows()

        # save tracked data
        #(test,grpHandle) = self.h5pyTools.getH5GroupName(self.f,'')
        #self.h5pyTools.createOverwriteDS(grpHandle,'angularSpeed',angularSpeed,['monitor',monitor])
        #self.h5pyTools.createOverwriteDS(grpHandle,'linearSpeed', linearSpeed)
        #self.h5pyTools.createOverwriteDS(grpHandle,'walkingTimes', wTimes, ['startTime',startTime])
        if not stopProgram:
            pickle.dump(frontpawPos, open(self.analysisLocation + '%s_%s_%s_frontpawLocations.p' % (mouse, date, rec), 'wb'))
            pickle.dump(hindpawPos, open(self.analysisLocation + '%s_%s_%s_hindpawLocations.p' % (mouse, date, rec), 'wb'))
            pickle.dump(rungs, open( self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec), 'wb' ) )
        return stopProgram, [Radius, xCenter, yCenter], [xPosition,yPosition], badVideo

    ########################################################################################################################
    # frontpawPos,hindpawPos,rungs,fTimes,angularSpeed,linearSpeed,sTimes
    def analyzePawsAndRungs(self,mouse,date,rec,frontpawPos,hindpawPos,rungs,fTimes,angularSpeed,linearSpeed,sTimes,angleTimes):
        # some image streams for verification
        self.showImages = False

        # create video output streams
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.outRung = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_rungAnalysis.avi' % (mouse, date, rec), fourcc, 20.0, (self.Vwidth, self.Vheight))
        # self.outPawRung  = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_pawRungTracking.avi' % (mouse, date, rec),fourcc, 20.0, (self.Vwidth, self.Vheight))



        # # wait and abort criterion, 'esc' allows to stop
        # k = cv2.waitKey(1) & 0xff
        # # print k
        # if k == 27: break

        ##########################################################
        # build array of paw positions
        def calculateDist(a,b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        spacingDegree = 6.81 #360./48.

        rungs = np.asarray(rungs)

        fp = []
        fpAll = []
        hp = []
        lastFrontPos = (frontpawPos[0][2][0],frontpawPos[0][2][1])
        fp.append([0, frontpawPos[0][0],frontpawPos[0][2][0],frontpawPos[0][2][1]])
        fD = []
        hD = []
        threshold = 60.
        skipSteps = 1
        for i in range(1,len(frontpawPos)-1):
            #currPos = (frontpawPos[i][2][0],frontpawPos[i][2][1])
            #nextPos = (frontpawPos[i+1][2][0],frontpawPos[i+1][2][1])
            #lastcurrDist = calculateDist(lastFrontPos,currPos)
            #currNextDist = calculateDist(currPos,nextPos)
            #fD.append([lastcurrDist,currNextDist])
            fp.append([i, frontpawPos[i][0],frontpawPos[i][2][0],frontpawPos[i][2][1]])
            # if (np.abs(lastcurrDist)+np.abs(currNextDist)) < threshold*skipSteps :
            #     fp.append([frontpawPos[i][0],frontpawPos[i][2][0],frontpawPos[i][2][1]])
            #     lastFrontPos = currPos
            #     skipSteps = 1
            # else:
            #     print 'fp', i, lastcurrDist, currNextDist
            #     skipSteps +=1
        lastHindPos = (hindpawPos[0][2][0],hindpawPos[0][2][1])
        hp.append([0,hindpawPos[0][0], hindpawPos[0][2][0], hindpawPos[0][2][1]])
        skipSteps=1
        for i in range(1,len(hindpawPos)):
            #currPos = (hindpawPos[i][2][0],hindpawPos[i][2][1])
            #nextPos = (hindpawPos[i+1][2][0],hindpawPos[i+1][2][1])
            #lastcurrDist = calculateDist(lastHindPos,currPos)
            #currNextDist = calculateDist(currPos, nextPos)
            #hD.append([lastcurrDist,currNextDist])
            hp.append([i, hindpawPos[i][0], hindpawPos[i][2][0], hindpawPos[i][2][1]])
            # if (np.abs(lastcurrDist)+np.abs(currNextDist)) < threshold*skipSteps :
            #     hp.append([hindpawPos[i][0], hindpawPos[i][2][0], hindpawPos[i][2][1]])
            #     lastHindPos = currPos
            #     skipSteps = 1
            # else:
            #     skipSteps +=1

        fp = np.asarray(fp)
        #fpAll = np.asarray(fpAll)
        hp = np.asarray(hp)
        ################################################################################
        # extract stance and swing phase through difference in speed

        def findBeginningAndEndOfStep(speedDiff,speedDiffThresh,minLength,trailingStart,trailingEnd):
            # determine regions during which the speed is different for more than xLength values
            thresholded = speedDiff > speedDiffThresh
            startStop = np.diff(np.arange(len(speedDiff))[thresholded]) > 1
            mmmStart = np.hstack((([True]), startStop))  # np.logical_or(np.hstack((([True]),startStop)),np.hstack((startStop,([True]))))
            mmmStop = np.hstack((startStop, ([True])))
            startIdx = (np.arange(len(speedDiff))[thresholded])[mmmStart]
            stopIdx = (np.arange(len(speedDiff))[thresholded])[mmmStop]
            minLengthThres = (stopIdx - startIdx) > minLength
            startStep = startIdx[minLengthThres]-trailingStart
            endStep = stopIdx[minLengthThres]+trailingEnd
            return np.column_stack((startStep,endStep))

        fpTimes = fTimes[np.array(fp[:,1],dtype=int)]
        hpTimes = fTimes[np.array(hp[:,1],dtype=int)]
        speedFP = (np.sqrt((np.diff(fp[:, 2]) / np.diff(fpTimes)) ** 2 + (np.diff(fp[:, 3]) / np.diff(fpTimes)) ** 2)) / 80.
        speedHP = (np.sqrt((np.diff(hp[:,2])/np.diff(hpTimes))**2 + (np.diff(hp[:,3])/np.diff(hpTimes))**2))/80.


        walk_interp = interp1d(sTimes, np.abs(linearSpeed))
        maskF = (fpTimes[1:]>=sTimes[0]) & (fpTimes[1:]<=sTimes[-1])
        newFPWheelSpeed = walk_interp(fpTimes[1:][maskF])
        fpSpeedDiff = speedFP[maskF]-newFPWheelSpeed
        maskH = (hpTimes[1:]>sTimes[0]) & (hpTimes[1:]<sTimes[-1])
        newHPWheelSpeed = walk_interp(hpTimes[1:][maskH])
        hpSpeedDiff = speedHP[maskH]-newHPWheelSpeed


        #pdb.set_trace()
        xLengthFP = 3
        xLengthHP = 4
        speedDiffThreshold = 10.
        startStopFPStep = findBeginningAndEndOfStep(fpSpeedDiff,5.,xLengthFP,2,3)
        startStopHPStep = findBeginningAndEndOfStep(hpSpeedDiff,speedDiffThreshold,xLengthHP,2,4)
        # join times of the respective frames
        startStopFPStep = np.column_stack((startStopFPStep,fpTimes[1:][maskF][startStopFPStep[:,0]],fpTimes[1:][maskF][startStopFPStep[:,1]]))
        startStopHPStep = np.column_stack((startStopHPStep,hpTimes[1:][maskH][startStopHPStep[:,0]],hpTimes[1:][maskH][startStopHPStep[:,1]]))
        #newAngles = walk_interp(fTimes[mask])
        #ax7.plot(sTimes, np.abs(linearSpeed))

        # plt.plot(fpTimes[1:][maskF],speedFP[maskF],c='0.5')
        # plt.plot(sTimes, np.abs(linearSpeed),c='C0')
        # for i in range(len(startStopFPStep)):
        #     #pdb.set_trace()
        #     plt.plot(fpTimes[1:][maskF][int(startStopFPStep[i,0]):int(startStopFPStep[i,1])],speedFP[maskF][int(startStopFPStep[i,0]):int(startStopFPStep[i,1])],c='C1')
        #
        # plt.show()
        # pdb.set_trace()
        ################################################################################
        # fit circle to points of rungscrews

        x = np.r_[rungs[:,1]]
        y = np.r_[rungs[:,2]]

        def calc_R(xc, yc):
            """ calculate the distance of each data points from the center (xc, yc) """
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        def f_2b(c):
            """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        def Df_2b(c):
            """ Jacobian of f_2b
            The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
            xc, yc = c
            df2b_dc = np.empty((len(c), x.size))

            Ri = calc_R(xc, yc)
            df2b_dc[0] = (xc - x) / Ri  # dR/dxc
            df2b_dc[1] = (yc - y) / Ri  # dR/dyc
            df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

            return df2b_dc

        center_estimate = [600,600]
        center_2b, ier = optimize.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)

        xc_2b, yc_2b = center_2b
        Ri_2b = calc_R(*center_2b)
        R_2b = Ri_2b.mean()
        pixToCmConversion = 12.5/R_2b
        print('Center, Radius of fitted circle : ', center_2b, R_2b, pixToCmConversion)

        ###########################################################
        # exclude points which are too far away from the circle fit line
        inclP = abs((Ri_2b - R_2b)) < 25
        rungs = rungs[inclP]

        ############################################################
        # fit list of points on a circle to the actual extracted points:
        # determine rotation angle and count rungs
        def angle_between(p1, p2):
            ang1 = np.arctan2(*p1[::-1])
            ang2 = np.arctan2(*p2[::-1])
            return np.rad2deg((ang1 - ang2) % (2 * np.pi))

        def contructCloudOfPointsOnCircle(nPoints,circleCenter,circleRadius,spacingDegree):
            cPoints = np.zeros((nPoints,3))
            for i in range(nPoints):
                cPoints[i,0] = i
                cPoints[i,1] = circleCenter[0] + np.sin(i*spacingDegree*np.pi/180.)*circleRadius
                cPoints[i,2] = circleCenter[1] - np.cos(i*spacingDegree*np.pi/180.)*circleRadius
            return cPoints



        def rotate_around_point(xy, degrees, origin=(0, 0)):
            """Rotate a point around a given point.
            """
            x, y = xy
            offset_x, offset_y = origin
            adjusted_x = (x - offset_x)
            adjusted_y = (y - offset_y)
            cos_rad = math.cos(degrees*np.pi/180.)
            sin_rad = math.sin(degrees*np.pi/180.)
            qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
            qy = offset_y - sin_rad * adjusted_x + cos_rad * adjusted_y

            return ([qx, qy])


        def fitfunc(d,wheelPoints,genericPoints):
            genericRotated = genericPoints.copy()
            distances = np.zeros(len(genericPoints))
            for i in range(len(genericPoints)):
                genericRotated[i] = rotate_around_point(genericPoints[i],d,origin=center_2b)
                distances[i] = calculateDist(genericRotated[i],wheelPoints[i])
            return np.sum(np.abs(distances))


        rungsList = rungs.tolist()
        rungsList.sort(key=lambda x: x[2])
        rungsList.sort(key=lambda x: x[1])
        rungsList.sort(key=lambda x: x[0])
        rungsSorted = np.asarray(rungsList)

        rungConvergencePoint = ([rungsSorted[0,3],rungsSorted[0,4]])
        # determine first angle in first image
        ppFFirst = rungsSorted[rungsSorted[:,0]==0][:,1:3]
        genericPs = contructCloudOfPointsOnCircle(len(ppFFirst),center_2b,R_2b,spacingDegree)
        d0 = -0.5
        d1, success = optimize.leastsq(fitfunc, d0,args=(ppFFirst,genericPs[:,1:]))

        # loop over all frames - via frameNumbers - not individual points
        frameNumbers = np.arange(len(fTimes)) #np.unique(rungsSorted[:,0])
        dOld = d1
        rungsNumbered = []
        rungCounter = 0
        #videoFileName = self.analysisLocation + '%s_%s_%s_pawRungTracking.avi' % (mouse, date, rec)
        #self.video = cv2.VideoCapture(videoFileName)
        #aBtw = []
        frontpawRungDist = []
        hindpawRungDist = []
        pC = np.zeros(8)
        for i in range(len(frameNumbers)):
            #ok, img = self.video.read()
            if self.showImages:
                blank_image = np.zeros((self.Vheight,self.Vwidth,3), np.uint8)

            # determine points per frame
            ppF = rungsSorted[rungsSorted[:,0]==frameNumbers[i]][:,1:]

            # for n in range(1,len(ppF)):
            #     #print angle_between(ppF[n]-center_2b,ppF[n-1]-center_2b),
            #     aBtw.append(angle_between(ppF[n]-center_2b,ppF[n-1]-center_2b))

            genericPs = contructCloudOfPointsOnCircle(len(ppF),center_2b,R_2b,spacingDegree)
            d1, success = optimize.leastsq(fitfunc, d0,args=(ppF[:,:2],genericPs[:,1:]))
            degreeDifference = d1-dOld
            if degreeDifference < -5.:
                rungCounter +=1
                degreeDifference += spacingDegree
            if degreeDifference > 5.:
                rungCounter -=1
                degreeDifference -= spacingDegree
            #dD.append(degreeDifference[0])
            numberedR = np.arange(len(ppF))+rungCounter
            rungsNumbered.append([i,frameNumbers[i],len(ppF),d1,degreeDifference[0],rungCounter,numberedR,ppF[:,:2]])

            # cacluate distance btw paw and rungs
            frontpawPos = fp[fp[:, 1] == frameNumbers[i]]
            hindpawPos = hp[hp[:, 1] == frameNumbers[i]]
            tempFD = []
            tempHD = []
            pC[len(ppF)] += 1
            if len(ppF)==1:
                nAddRungs = 3
            elif len(ppF)==2 or len(ppF)==3:
                nAddRungs = 2
            elif len(ppF)==4 or len(ppF)==5:
                nAddRungs = 1
            elif len(ppF)==6 or len(ppF)==7:
                nAddRungs = 0
            genericPs = contructCloudOfPointsOnCircle(7,center_2b,R_2b,spacingDegree)
            numberedGenericR = np.arange(7) + numberedR[0] - nAddRungs
            genericRotated = genericPs[:,1:].copy()
            #pdb.set_trace()
            for n in range(len(genericPs)):
                genericRotated[n] = rotate_around_point(genericPs[n][1:],d1+nAddRungs*spacingDegree,origin=center_2b)
                #print frameNumbers[i], len(ppF), ppF
                #for n in range(len(ppF)):
                if len(frontpawPos)>0:
                    tempFD.append(np.cross(rungConvergencePoint-genericRotated[n],frontpawPos[0][2:]-genericRotated[n])/norm(rungConvergencePoint-genericRotated[n]))
                if len(hindpawPos)>0:
                    tempHD.append(np.cross(rungConvergencePoint-genericRotated[n],hindpawPos[0][2:]-genericRotated[n])/norm(rungConvergencePoint-genericRotated[n]))
                    #hindpawRungDist.append(tempD[1])
                if self.showImages:
                    cv2.circle(blank_image, (int(genericRotated[n,0]), int(genericRotated[n,1])), 6, (0, 0, 255), 3)
            if len(frontpawPos)>0:
                #tempAFD = np.asarray(tempFD)
                #sortedA = np.argsort(tempAFD)
                #pdb.set_trace()
                frontpawRungDist.append([i,frameNumbers[i]])
                frontpawRungDist[-1].extend(tempFD)
                frontpawRungDist[-1].extend(numberedGenericR.tolist())
            temp = []
            if len(hindpawPos)>0:
                #tempAHD = np.asarray(tempHD)
                #sortedA = np.argsort(tempAHD)
                hindpawRungDist.append([i,frameNumbers[i]])
                hindpawRungDist[-1].extend(tempHD)
                hindpawRungDist[-1].extend(numberedGenericR.tolist())
            if self.showImages:
                for n in range(len(ppF)):
                    cv2.circle(blank_image, (ppF[n,0], ppF[n,1]), 2, (0, 255, 0), 2)
            # draw the center of the circle
            # cv2.circle(blank_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            # self.outRung.write(orig)

            if self.showImages:
                cv2.imshow("Rung rotations : %s   rec : %s/%s" % (mouse, date, rec), blank_image)

            dOld = d1
            #for n in range(len(ppF)):
            # if i in [836,837,838]:
            #     print i
            #     for n in range(len(ppF)):
            #         cv2.putText(img,'%s' % (n+rungCounter),(ppF[n,0],ppF[n,1]),cv2.FONT_HERSHEY_SIMPLEX,2,color=(0, 0, 255))
            #
            #     cv2.imshow("Paw-tracking monitor", img)
            #     # wait and abort criterion, 'esc' allows to stop
            if self.showImages:
                k = cv2.waitKey(50) & 0xff
                if k == 27: break
                elif k == 32: pdb.set_trace()
            #cv2.destroyAllWindows()

            #if degreeDifference[0] > 1:
            #    print i,frameNumbers[i], len(ppF), d1, degreeDifference, np.arange(len(ppF))+rungCounter #, ppF

            #if i == 30 :
            #    pdb.set_trace()
        print('rung number per image : ', pC)
        #pdb.set_trace()
        frontpawRungDist = np.asarray(frontpawRungDist)
        hindpawRungDist = np.asarray(hindpawRungDist)
        ###################################################################
        # substract rotation

        if len(frameNumbers) != len(fTimes):
            print('problem with frame number length')
            pdb.set_trace()

        #pdb.set_trace()
        walk_interp = interp1d(angleTimes[:,0],angleTimes[:,1])

        mask = (fTimes>angleTimes[:,0][0]) & (fTimes<angleTimes[:,0][-1])

        newAngles = walk_interp(fTimes[mask])
        # # sTimes, ttime
        # ax01.plot(angleTimes[:,0],linearSpeed)
        # ax01.plot(ttime[mask],newWalking)
        #
        degreesTurned = 0.
        fpLinear = []
        hpLinear = []
        #pdb.set_trace()
        fInitial = fp[0][2:]
        hInitial = hp[0][2:]
        rotationsHP = 0.
        rotationsFP = 0.
        oldAfp = 0.
        oldAhp = 0.
        for i in range(len(fTimes[mask])):
            fpMask = fp[:,1]==frameNumbers[i]
            hpMask = hp[:,1]==frameNumbers[i]

            degreesTurned += rungsNumbered[i][4]
            #pdb.set_trace()
            rfp = rotate_around_point(fp[fpMask][:,2:][0],-newAngles[i],center_2b)
            rhp = rotate_around_point(hp[hpMask][:,2:][0],-newAngles[i],center_2b)

            afp = angle_between(rfp-center_2b,fInitial-center_2b)
            if oldAfp > 300. and afp < 100. :
                rotationsFP +=1.
            elif oldAfp < 100. and afp > 300. :
                rotationsFP -=1.
            dfp = (rotationsFP + afp/360.)*80.
            ahp = angle_between(rhp-center_2b,hInitial-center_2b)
            if oldAhp > 300. and ahp < 100. :
                rotationsHP +=1.
            elif oldAhp < 100. and ahp > 300. :
                rotationsHP -=1.
            dhp = (rotationsHP + ahp/360.)*80.
            # rotational coordinates to straight motion : distance is y
            fpLinear.append([frameNumbers[i],fTimes[i],dfp,(calculateDist(rfp,center_2b)-R_2b)*pixToCmConversion,newAngles[i],degreesTurned])
            hpLinear.append([frameNumbers[i],fTimes[i],dhp,(calculateDist(rhp,center_2b)-R_2b)*pixToCmConversion,newAngles[i],degreesTurned])
            print(i,degreesTurned,afp,dfp,ahp,dhp, rotationsFP, rotationsHP,newAngles[i])
            oldAfp = afp
            oldAhp = ahp

        #cpdb.set_trace()
        return (fp,hp,rungs,center_2b, R_2b,rungsNumbered,np.asarray(fpLinear),np.asarray(hpLinear),frontpawRungDist,hindpawRungDist,startStopFPStep,startStopHPStep)


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
import time


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


class pawClassifier:
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
        print 'on exit'

    ############################################################
    def extratContourInformation(self,mouse,date,rec, thresholdPawList, wm):
        # tracking parameters #########################
        self.thresholdValue = 0.7 # in %
        self.minContourArea = 40 # square pixels

        self.scoreWeights = {'distanceWeight':5,'areaWeight':1}
        ###############################################

        videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior.avi' % (mouse, date, rec)

        self.video = cv2.VideoCapture(videoFileName)
        self.Vlength = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.Vwidth = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Vheight = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.Vfps = self.video.get(cv2.CAP_PROP_FPS)

        print 'Video properties : %s frames, %s pixels width, %s pixels height, %s fps' % (self.Vlength, self.Vwidth, self.Vheight, self.Vfps)
        if not self.video.isOpened():
            print "Could not open video"
            sys.exit(1)

        if self.Vlength != len(thresholdPawList):
            print 'list of thresholds does not correspond to length of video'
            sys.exit(1)



        # create video output streams
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.outPaw = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_pawTracking.avi' % (mouse, date, rec), fourcc, 20.0, (self.Vwidth, self.Vheight))
        # self.outPawRung  = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_pawRungTracking.avi' % (mouse, date, rec),fourcc, 20.0, (self.Vwidth, self.Vheight))

        # read first video frame
        # ok, img = self.video.read()
        # if not ok:
        #     print 'Cannot read video file'
        #     sys.exit()

        
        # Return an array representing the indices of a grid.
        imgGrid = np.indices((self.Vheight, self.Vwidth))
        wheelMask = np.zeros((self.Vheight, self.Vwidth))

        mask = np.sqrt((imgGrid[1] - wm[1]) ** 2 + (imgGrid[0] - wm[2]) ** 2) > wm[0]
        #maskInv = np.sqrt((imgGrid[1] - xCenter) ** 2 + (imgGrid[0] - yCenter) ** 2) < Radius
        wheelMask[mask] = 1
        #wheelMaskInv[maskInv] = 1
        wheelMask = np.array(wheelMask, dtype=np.uint8)

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

        allContours = []
        frontpawPos = []
        hindpawPos = []
        hcheckPos = -1
        nF = 0
        #########################################################################
        # loop over all images in video
        print("Mouse : %s\nRecording : %s // %s" % (mouse, date, rec))
        print ("Running the tracking algorithm...")
        while True:
            #os.system('clear')
            #print("Mouse : %s\nRecording : %s // %s" % (mouse, date, rec))
            # Read a new frame
            thresholdV = self.thresholdValue
            ok, img = self.video.read()
            if not ok:
                break
            orig = img.copy()
            #origCL = img.copy()

            imgMouse = cv2.bitwise_and(img, img, mask=wheelMask)

            # convert image to gray-scale
            imgGMouse = cv2.cvtColor(imgMouse, cv2.COLOR_BGR2GRAY)

            #################################################################################################
            # find contours based on threshold value

            # blur image and apply threshold
            blur = cv2.GaussianBlur(imgGMouse, (5, 5), 0)
            minMaxL = cv2.minMaxLoc(blur)

            ret, th1 = cv2.threshold(blur, minMaxL[1] * thresholdPawList[nF,0], 255, cv2.THRESH_BINARY)
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

            #print mmean, sstddev
            #for c in cnts:
            #    mask = np.zeros(imgGMouse.shape, dtype="uint8")
            #    cv2.drawContours(mask, c, 0, 255, 2)
            #    mmean, sstddev = cv2.meanStdDev(imgGMouse, mask=mask)
            #    print cv2.contourArea(c), mmean, sstddev
            #pdb.set_trace()
            cv2.drawContours(orig, cnts, -1, (0, 255, 0), 2)

            #cntDistances = []
            statsRois = []
            contourProperties = []
            nCtr = 0

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
                #cv2.drawContours(mask, c, -1, (255), 1)  # cv.drawContours(mask, contours, -1, (255),1)
                mmean, sstddev = cv2.meanStdDev(imgGMouse, mask=mask)
                #statsRois.append([cv2.contourArea(c), mmean[0][0], sstddev[0][0]])

                #rois.append([ell,np.pi * (ell[1][0] / 2.) * (ell[1][1] / 2.0)])


                # cornerDist.append(dist.euclidean((0, self.Vheight), ell[0]))
                if nF > 0:
                    #pdb.set_trace()
                    frontDist = dist.euclidean((frontpawPos[-1][2],frontpawPos[-1][3]), ell[0])
                    hindDist = dist.euclidean((hindpawPos[-1][2],hindpawPos[-1][3]), ell[0])
                else:
                    frontDist = 0.
                    hindDist  = 0.
                #frontAreaChange.append(np.abs(frontpawPos[fcheckPos][2] - statsRois[i][0]))
                #hindAreaChange.append(np.abs(hindpawPos[hcheckPos][2] - statsRois[i][0]))

                # nFrame, nContour, area, mean of contour, STD of contour,
                contourProperties.append([nF,nCtr,ell[0][0],ell[0][1],cv2.contourArea(c),mmean[0][0],sstddev[0][0],frontDist,hindDist])
                if nCtr == thresholdPawList[nF,1]:
                    orig = cv2.ellipse(orig, ell, (0, 255, 0), 2)
                if nCtr == thresholdPawList[nF,2]:
                    orig = cv2.ellipse(orig, ell, (0, 0, 255), 2)
                nCtr += 1

            # if nCtr == thresholdPawList[nF,1]:
            # nFrame, nContour, x, y, area
            fpIdx = int(thresholdPawList[nF,1])
            frontpawPos.append([nF,thresholdPawList[nF,1],contourProperties[fpIdx][2],contourProperties[fpIdx][3],contourProperties[fpIdx][4]])
            # elif nCtr == thresholdPawList[nF,2]:
            hpIdx = int(thresholdPawList[nF,2])
            hindpawPos.append([nF,thresholdPawList[nF,1],contourProperties[hpIdx][2],contourProperties[hpIdx][3],contourProperties[hpIdx][4]])



            allContours.append(contourProperties)
            #pdb.set_trace()
            #cv2.rectangle(orig, (0, self.Vheight), (x, self.Vheight - 15), (100, 100, 100), thickness=-1)
            #cv2.putText(orig, FrameStr, (0, self.Vheight - 20), cv2.QT_FONT_NORMAL, 0.45, color=(255, 255, 255))
            #cv2.putText(orig, Str, (0, self.Vheight - 5), cv2.QT_FONT_NORMAL, 0.4, color=(220, 220, 220))
            #cv2.putText(orig, Str2, (int(self.Vwidth / 3), self.Vheight - 5), cv2.QT_FONT_NORMAL, 0.4, color=(220, 220, 220))
            cv2.putText(orig, 'frontpaw', (10, 30), cv2.QT_FONT_NORMAL, 0.7, color=(0, 255, 0))
            #orig = cv2.ellipse(orig, rois[frontIdx][0], (0, 255, 0), 2)
            cv2.putText(orig, 'hindpaw', (10, 60), cv2.QT_FONT_NORMAL, 0.7, color=(0, 0, 255))
            #orig = cv2.ellipse(orig, rois[hindIdx][0], (0, 0, 255), 2)
            # print 'hind ',


            #self.outPaw.write(orig)

            if self.showImages:
                cv2.imshow("Paw-tracking monitor - mouse : %s   rec : %s/%s" % (mouse, date, rec), orig)

            # wait and abort criterion, 'esc' allows to stop
            k = cv2.waitKey(10) & 0xff
            #print k
            if k == 27: break
            elif k == 32: # space key stops the
                pdb.set_trace()
                #cv2.waitKey(100)
            nF += 1
            print nF, len(cnts)

        cv2.destroyAllWindows()

        # save tracked data
        #(test,grpHandle) = self.h5pyTools.getH5GroupName(self.f,'')
        #self.h5pyTools.createOverwriteDS(grpHandle,'angularSpeed',angularSpeed,['monitor',monitor])
        #self.h5pyTools.createOverwriteDS(grpHandle,'linearSpeed', linearSpeed)
        #self.h5pyTools.createOverwriteDS(grpHandle,'walkingTimes', wTimes, ['startTime',startTime])
        pickle.dump(frontpawPos, open(self.analysisLocation + '%s_%s_%s_correctFrontpawLocations.p' % (mouse, date, rec), 'wb'))
        pickle.dump(hindpawPos, open(self.analysisLocation + '%s_%s_%s_correctHindpawLocations.p' % (mouse, date, rec), 'wb'))
        pickle.dump(allContours, open( self.analysisLocation + '%s_%s_%s_contourInformation.p' % (mouse, date, rec), 'wb' ) )
        #return stopProgram, [Radius, xCenter, yCenter], [xPosition,yPosition], badVideo
    ########################################################
    def click_and_crop(self, event, x, y, flags, param):
        # grab references to the global variables
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouseX = x
            self.mouseY = y
            self.success = True
            self.clicked = True
            #print x,y
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.success = False
            self.clicked = True
    ############################################################
    def createTrainingSetWithFeedback(self, mouse, date, rec, wm,startStopFrames = None):
        # tracking parameters #########################
        self.thresholdValue = 0.26  # in %
        self.minContourArea = 30  # square pixels


        hpClicks = []
        ###############################################
        # loop over front and hindpaw position
        for paw in ['hindpaw']: #['frontpaw','hindpaw']:

            clicks = []
            if paw == 'frontpaw':
                pawID = 0
            elif paw == 'hindpaw':
                pawID = 1

            videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior.avi' % (mouse, date, rec)

            self.video = cv2.VideoCapture(videoFileName)
            self.Vlength = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.Vwidth = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.Vheight = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.Vfps = self.video.get(cv2.CAP_PROP_FPS)

            if startStopFrames == None:
                startFrame = 0
                endFrame = self.Vlength
            else:
                startFrame = startStopFrames[0]
                endFrame = startStopFrames[1]

            print 'Video properties : %s frames, %s pixels width, %s pixels height, %s fps' % (self.Vlength, self.Vwidth, self.Vheight, self.Vfps)
            if not self.video.isOpened():
                print "Could not open video"
                sys.exit(1)

            # create video output streams
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # self.outPaw = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_pawTracking.avi' % (mouse, date, rec), fourcc, 20.0, (self.Vwidth, self.Vheight))
            # self.outPawRung  = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_pawRungTracking.avi' % (mouse, date, rec),fourcc, 20.0, (self.Vwidth, self.Vheight))

            # read first video frame
            # ok, img = self.video.read()
            # if not ok:
            #     print 'Cannot read video file'
            #     sys.exit()

            # Return an array representing the indices of a grid.
            imgGrid = np.indices((self.Vheight, self.Vwidth))
            wheelMask = np.zeros((self.Vheight, self.Vwidth))
            #pdb.set_trace()
            # exclude area of rung screws and the upper left corner where time and frame numbers are displayed
            mask = (np.sqrt((imgGrid[1] - wm[1]) ** 2 + (imgGrid[0] - wm[2]) ** 2) > wm[0] ) & ((imgGrid[0]>50)&(imgGrid[1]>80))
            # maskInv = np.sqrt((imgGrid[1] - xCenter) ** 2 + (imgGrid[0] - yCenter) ** 2) < Radius
            wheelMask[mask] = 1
            # wheelMaskInv[maskInv] = 1
            wheelMask = np.array(wheelMask, dtype=np.uint8)

            nF = 0
            #########################################################################
            # loop over all images in video

            #print("Mouse : %s\nRecording : %s // %s" % (mouse, date, rec))
            #print ("Running the tracking algorithm...")
            while True:
                # os.system('clear')
                # print("Mouse : %s\nRecording : %s // %s" % (mouse, date, rec))
                # Read a new frame
                thresholdV = self.thresholdValue
                self.clicked = False
                ok, img = self.video.read()
                if not ok:
                    break
                if nF not in range(startFrame,endFrame):
                    continue
                orig = img.copy()
                #origCL = img.copy()
                # orig3 = img.copy()
                # while (1):
                # ret, frame = cap.read()

                # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #imgInv = cv2.bitwise_and(img, img, mask=wheelMaskInv)
                imgMouse = cv2.bitwise_and(img, img, mask=wheelMask)

                # convert image to gray-scale
                #imgGWheel = cv2.cvtColor(imgInv, cv2.COLOR_BGR2GRAY)
                imgGMouse = cv2.cvtColor(imgMouse, cv2.COLOR_BGR2GRAY)

                #################################################################################################
                # find contours based on maximal illumination

                # blur image and apply threshold
                blur = cv2.GaussianBlur(imgGMouse, (5, 5), 0)
                minMaxL = cv2.minMaxLoc(blur)
                # mask = np.zeros(imgGMouse.shape,dtype="uint8")
                # cv2.drawContours(mask, [contour], -1, 255, -1)
                # mean,stddev = cv2.meanStdDev(image,mask=mask)
                #while True:
                ret, th1 = cv2.threshold(blur, minMaxL[1] * thresholdV, 255, cv2.THRESH_BINARY)
                # print ret, th1
                # perform edge detection, then perform a dilation + erosion to
                # close gaps in between object edges
                edged = cv2.Canny(th1, 50, 100)
                edged = cv2.dilate(edged, None, iterations=1)
                edged = cv2.erode(edged, None, iterations=1)

                cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                #cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
                #largeCnts = (c if cv2.contourArea(c) for c in cntsSorted)
                #pdb.set_trace()
                if len(cnts) > 0:
                    # sort the contours from left-to-right and initialize the
                    # 'pixels per metric' calibration variable
                    (cnts, _) = contours.sort_contours(cnts)
                #nLarge = 0
                cLarge = []
                for c in cnts:
                    if cv2.contourArea(c) > self.minContourArea:
                        cLarge.append(c)
                #for c in cnts:
                #    if cv2.contourArea(c) > self.minContourArea:
                #        nLarge += 1
                #if len(cLarge) >=5:
                #    break
                #else:
                #    thresholdV = thresholdV - 0.05
                # print mmean, sstddev
                # for c in cnts:
                #    mask = np.zeros(imgGMouse.shape, dtype="uint8")
                #    cv2.drawContours(mask, c, 0, 255, 2)
                #    mmean, sstddev = cv2.meanStdDev(imgGMouse, mask=mask)
                #    print cv2.contourArea(c), mmean, sstddev
                # pdb.set_trace()
                cv2.drawContours(orig, cLarge ,-1, (0,255,0), 2)
                # cntDistances = []
                statsRois = []
                rois = []
                for c in cLarge:
                    # if the contour is not sufficiently large, ignore it
                    #if cv2.contourArea(c) < self.minContourArea:
                    #    continue

                    # print 'contourArea : ',cv2.contourArea(c)
                    # compute the rotated bounding box of the contour
                    ell = cv2.fitEllipse(c)
                    # print ell
                    # cntDistances.append(dist.euclidean(pawPos[checkPos][3], ell[0]))
                    # cntArea.append(np.pi * (ell[1][0] / 2.) * (ell[1][1] / 2.0))
                    # get statistics of contour
                    #mask = np.zeros(imgGMouse.shape, dtype="uint8")
                    #cv2.drawContours(mask, c, -1, (255), 1)  # cv.drawContours(mask, contours, -1, (255),1)
                    #mmean, sstddev = cv2.meanStdDev(imgGMouse, mask=mask)
                    #statsRois.append([cv2.contourArea(c), mmean[0][0], sstddev[0][0]])

                    rois.append([ell])
                    cv2.ellipse(orig, ell, (255, 0, 0), 2)
                if paw == 'frontpaw':
                    cv2.putText(orig,'left click on frontpaw ROI',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,color=(0, 255, 0))
                    cv2.putText(orig,'middle click if NO frontpaw ROI',(10,77),cv2.FONT_HERSHEY_SIMPLEX,0.6,color=(0, 255, 0))
                elif paw == 'hindpaw':
                    cv2.putText(orig,'left click on hindpaw ROI',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,color=(255, 255, 0))
                    cv2.putText(orig,'middle click NO hindpaw ROI',(10,77),cv2.FONT_HERSHEY_SIMPLEX,0.6,color=(255, 255, 0))
                #
                if self.showImages:
                    cv2.namedWindow("image",1)
                    cv2.setMouseCallback("image", self.click_and_crop)
                    cv2.imshow("image" , orig)
                    #cv2.imshow("Paw-tracking monitor - mouse : %s   rec : %s/%s" % (mouse, date, rec), orig)

                # wait and abort criterion, 'esc' allows to stop

                #print k
                #if k == 27: break
                #elif k == 32:
                #    pdb.set_trace()
                    #cv2.waitKey(100)
                while True:
                    if self.clicked: break
                    k = cv2.waitKey(1) & 0xff
                    if k == 27: break
                    #time.sleep(0.1)
                if k == 27: break
                ellClickDistances = np.ones(len(rois))*1000.
                for n in range(len(rois)):
                    ellClickDistances[n] = dist.euclidean((self.mouseX,self.mouseY), rois[n][0][0])

                NfpRois = np.argmin(ellClickDistances)
                # elif k == 32:
                #    pdb.set_trace()
                # cv2.waitKey(100)
                print nF, pawID, int(self.success), thresholdV, len(rois), self.mouseX, self.mouseY, NfpRois, rois[NfpRois][0][0][0], rois[NfpRois][0][0][1]
                clicks.append([nF, pawID, int(self.success), thresholdV, len(rois), self.mouseX, self.mouseY, NfpRois, rois[NfpRois][0][0][0], rois[NfpRois][0][0][1]])
                #print nF,self.mouseX,self.mouseY
                nF += 1
                #if nF==1000:
                #    break

            self.video.release()
            cv2.destroyAllWindows()
            if paw == 'frontpaw':
                pickle.dump(clicks, open(self.analysisLocation + '%s_%s_%s_supervisedFrontpawPositions.p' % (mouse, date, rec), 'wb'))
            elif paw == 'hindpaw':
                pickle.dump(clicks, open(self.analysisLocation + '%s_%s_%s_supervisedHindpawPositions.p' % (mouse, date, rec), 'wb'))
import cv2
import sys
import pdb
import pickle
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
from scipy.spatial import distance as dist

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


class openCVImageProcessingTools:
    def __init__(self,analysisLoc, figureLoc, ff, showI = False):
        #  "/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/170606_f37/170606_f37_2017.07.12_001_behavingMLI_000_raw_behavior.avi"
        self.analysisLocation = analysisLoc
        self.figureLocation = figureLoc
        self.f = ff
        self.showImages = showI


    ############################################################
    def __del__(self):
        self.video.release()

        cv2.destroyAllWindows()
        print 'on exit'

    ############################################################
    def trackPawsAndRungs(self,mouse,date,rec):

        def decideonAndAddPawPositions(pawPos,checkPos,rois):
            cntDistances = []
            cntArea = []
            #pdb.set_trace()
            for i in range(len(rois)):
                cntDistances.append(dist.euclidean(pawPos[checkPos][2], rois[i][0][0]))
                cntArea.append(rois[i][1])
            # lockBack = (3 if len(pawPos)>=3 else len(pawPos))
            # print lockBack
            # pdb.set_trace()
            # pd = [pawPos[i][3] for i in range(-lockBack,0)]
            # print pd
            # Aslope, Aintercept,_,_,_ = stats.linregress(range(lockBack),[pawPos[i][3] for i in range(-lockBack,0)])
            # Dslope, Dintercept,_,_,_ = stats.linregress(range(lockBack),[pawPos[i][2] for i in range(-lockBack,0)])
            # Aprojection = (lockBack+1.)*Aslope + Aintercept
            # Dprojection = (lockBack+1.)*Dslope + Dintercept
            # Aindex = np.argmin(abs(np.asarray(cntArea)-Aprojection))
            Dchange = abs(np.asarray(cntDistances))
            Achange = abs(np.asarray(cntArea) / pawPos[checkPos][3] - 1.) * 100.  # in percent
            DWeight = 0.9
            #print checkPos, Dchange, Achange,
            Pindex = np.argmin(DWeight * Dchange + (1. - DWeight) * Achange)
            # Dindex = np.argmin(abs(np.asarray(cntDistances)-Dprojection))
            # print Dindex, Dprojection, pawPos[-2:]
            # if abs(cntArea[Dindex]/pawPos[-1][3] - 1.) < 0.2:
            # print cntDistances[Aindex]
            if (Dchange[Pindex] < abs(checkPos) * 60.) and (Achange[Pindex] < abs(checkPos) * 200.):
                print 'success', Pindex, Dchange[Pindex], Achange[Pindex],
                return (0,[nF, rois[Pindex][0], rois[Pindex][0][0], cntArea[Pindex]])
                #pawPos.append([nF, rois[Pindex], rois[Pindex][0], cntArea[Pindex]])

                # orig2 = cv2.ellipse(orig2, rois[Pindex], (0, 255, 0), 2)
                # orig3 = cv2.ellipse(orig3, rois[Pindex], (0, 255, 0), 2)
                checkPos = -1  # pointLoc = rois[Dindex][0]  # maxStepCurrent = maxStep
            else:
                print 'failure', Pindex, Dchange[Pindex], Achange[Pindex],
                return (1,[nF,-1, -1, -1])
                #pawPos.append([nF, '', -1, -1, -1])
                #checkPos -= 1

        #################################################################################

        # tracking parameters #########################
        self.thresholdValue = 0.7 # in %
        self.minContourArea = 40 # square pixels
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
            sys.exit()

        # create video output streams
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.outPaw = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_pawTracking.avi' % (mouse, date, rec), fourcc, 20.0, (self.Vwidth, self.Vheight))
        self.outPawRung  = cv2.VideoWriter(self.analysisLocation + '%s_%s_%s_pawRungTracking.avi' % (mouse, date, rec),fourcc, 20.0, (self.Vwidth, self.Vheight))

        # read first video frame
        ok, img = self.video.read()
        if not ok:
            print 'Cannot read video file'
            sys.exit()

        recs = []
        bboxFront = cv2.selectROI("Select dot for FRONT paw", img, False)
        print recs
        bboxHind = cv2.selectROI("Select dot for HIND paw", img, False)
        cv2.destroyAllWindows()
        print 'front, hind paw bounding boxes : ', bboxFront, bboxHind
        pointLoc = bboxFront[:2]
        print 'bounding box area : ', bboxFront[2] * bboxFront[3]

        # Return an array representing the indices of a grid.
        imgGrid = np.indices((self.Vheight, self.Vwidth))

        Radius = 1500 # 1400
        xCenter = 1205 #1485
        yCenter = 1625 #1545

        ########################################################################
        # loop to find correct wheel mask
        nIt = 0
        while True:

            imgCircle = img.copy()
            cv2.circle(imgCircle, (xCenter, yCenter), Radius, (0, 0, 255), 2)
            if nIt > 0:
                cv2.circle(imgCircle, (oldxCenter, oldyCenter), oldRadius, (0, 0, 100), 2)
                #cv2.putText(imgCircle,'now',(10,10),color=(0, 0, 255))
                #cv2.putText(imgCircle,'before',(10,20),fontScale=4,color=(0, 0, 150),thickness=2)
            cv2.imshow("Wheel mask %s" % nIt , imgCircle)
            cv2.waitKey(1000)
            print 'current radius, xCenter, yCenter : ' , Radius, xCenter, yCenter
            var = raw_input("Enter new radius, xCenter, yCenter coordinates (separated by commas), otherwise press 'k' : ")
            #print "you entered", var
            if var == 'k':
                cv2.destroyWindow("Wheel mask %s" % (nIt))
                break
            else:
                (oldRadius,oldxCenter,oldyCenter) = (Radius, xCenter, yCenter)
                (Radius, xCenter, yCenter) = map(int, var.split(','))
                nIt += 1
                cv2.destroyWindow("Wheel mask %s" % (nIt-1))
                #if nIt >= 2:
                #    cv2.destroyWindow("Wheel mask %s" % (nIt-1))

        print 'masking after loop, Radius = %s, xCenter %s, yCenter = %s' % (Radius, xCenter, yCenter)

        wheelMask = np.zeros((self.Vheight, self.Vwidth))
        wheelMaskInv = np.zeros((self.Vheight, self.Vwidth))

        # create masks for mouse area and for lower area
        mask = np.sqrt((imgGrid[1] - xCenter) ** 2 + (imgGrid[0] - yCenter) ** 2) > Radius
        maskInv = np.sqrt((imgGrid[1] - xCenter) ** 2 + (imgGrid[0] - yCenter) ** 2) < Radius
        wheelMask[mask] = 1
        wheelMaskInv[maskInv] = 1
        wheelMask = np.array(wheelMask, dtype=np.uint8)
        wheelMaskInv = np.array(wheelMaskInv, dtype=np.uint8)

        # print wheelMask
        maxStep = 40
        maxStepCurrent = maxStep
        nF = 1
        rungs = []
        frontpawPos = []
        hindpawPos = []
        # append first paw postions to list : number of image, idendity 'front' or 'hind', , position, area
        frontpawPos.append([0, bboxFront[:2], [], np.pi * bboxFront[2] * bboxFront[3] / 4.])
        hindpawPos.append([0, bboxHind[:2], [], np.pi * bboxHind[2] * bboxHind[3] / 4.])
        #hindPawPos.append([0, [], bboxHind[:2], np.pi * bboxHind[2] * bboxHind[3] / 4.])
        fcheckPos = -1
        hcheckPos = -1
        # loop over all images in video
        while True:
            # Read a new frame
            thresholdV = self.thresholdValue
            ok, img = self.video.read()
            if not ok:
                break
            orig = img.copy()
            #orig2 = img.copy()
            #orig3 = img.copy()
            # while (1):
            # ret, frame = cap.read()

            # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgInv = cv2.bitwise_and(img, img, mask=wheelMaskInv)
            imgMouse = cv2.bitwise_and(img, img, mask=wheelMask)

            ###############################################################################################
            # find location of rungs
            # convert image to gray-scale
            imgGWheel = cv2.cvtColor(imgInv, cv2.COLOR_BGR2GRAY)
            # find circles in the lower part of the image, i.e., find screws to determine paw positions,
            circles = cv2.HoughCircles(imgGWheel, cv2.HOUGH_GRADIENT, 1, minDist=22, param1=50, param2=20, minRadius=30, maxRadius=40)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(imgInv, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv2.circle(imgInv, (i[0], i[1]), 2, (0, 0, 255), 3)
                    #cv2.circle(orig3, (i[0], i[1]), 2, (0, 0, 255), 3)
                    #cv2.line(imgInv, (i[0], i[1]), (500, -20), (255, 0, 0), 3)
                    rungs.append([nF, i[0], i[1], 500, -20])
                if self.showImages:
                    cv2.imshow('detected circles', imgInv)

            #################################################################################################
            # find contours based on maximal illumination

            # blur image and apply threshold
            imgGMouse = cv2.cvtColor(imgMouse, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(imgGMouse, (5, 5), 0)
            minMaxL = cv2.minMaxLoc(blur)
            while True:
                ret, th1 = cv2.threshold(blur, minMaxL[1] * thresholdV, 255, cv2.THRESH_BINARY)
                # print ret, th1
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
                if nLarge >= 2 :
                    break
                else:
                    thresholdV = thresholdV - 0.05
            # print cnts
            #pdb.set_trace()

            #cntDistances = []
            #cntArea = []
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
                rois.append([ell,np.pi * (ell[1][0] / 2.) * (ell[1][1] / 2.0)])
                orig = cv2.ellipse(orig, ell, (255, 0, 0), 2)

            # find ellipse which is the best continuation of the previous ones
            # print 'nContours, Dist, Areaa : ', nCnts, cntDistances, cntArea
            print 'frame ', nF, len(rois),
            # if rois were detected
            if len(rois) > 0:
                cornerDist = []
                frontDist  = []
                hindDist   = []
                #pdb.set_trace()
                for i in range(len(rois)):
                    cornerDist.append(dist.euclidean((0,self.Vheight), rois[i][0][0]))
                    frontDist.append(dist.euclidean(frontpawPos[-1][1], rois[i][0][0]))
                    hindDist.append(dist.euclidean(hindpawPos[-1][1], rois[i][0][0]))
                if len(cornerDist) == 2:
                    hindIdx =  np.argmin(np.asarray(cornerDist))
                    frontIdx = np.argmax(np.asarray(cornerDist))
                else :
                    hindIdx = np.argmin(np.asarray(hindDist))
                    frontIdx = np.argmin(np.asarray(frontDist))
                print 'front, hind index ', frontIdx, hindIdx
                #pdb.set_trace()
                frontpawPos.append([nF, rois[frontIdx][0][0],rois[frontIdx], 0.1 ])
                hindpawPos.append([nF, rois[hindIdx][0][0], rois[frontIdx], 0.1] )
                ##
                orig = cv2.ellipse(orig, rois[frontIdx][0], (0, 255, 0), 2)
                #print 'hind ',
                #dret = decideonAndAddPawPositions(hindpawPos, hcheckPos, rois)
                #hindpawPos.append(dret[1])
                #hcheckPos += dret[0]
                #if not dret[0]:
                orig = cv2.ellipse(orig, rois[hindIdx][0], (0, 0, 255), 2)
                # pdb.set_trace()
                print '\n'
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
                print 'failure no rois'
                frontpawPos.append([nF, -1, -1])
                hindpawPos.append([nF,-1, -1])
                fcheckPos -= 1
                hcheckPos -= 1
            # show image with all detected rois, and rois decided to be paws
            if self.showImages:
                cv2.imshow("Image", orig)

            # wait and abort criterion, 'esc' allows to stop
            k = cv2.waitKey(1) & 0xff
            if k == 27: break

            nF += 1

        cv2.destroyAllWindows()

        # save tracked data
        #(test,grpHandle) = self.h5pyTools.getH5GroupName(self.f,'')
        #self.h5pyTools.createOverwriteDS(grpHandle,'angularSpeed',angularSpeed,['monitor',monitor])
        #self.h5pyTools.createOverwriteDS(grpHandle,'linearSpeed', linearSpeed)
        #self.h5pyTools.createOverwriteDS(grpHandle,'walkingTimes', wTimes, ['startTime',startTime])

        pickle.dump(frontpawPos, open(self.analysisLocation + '%s_%s_%s_frontpawLocations.p' % (mouse, date, rec), 'wb'))
        pickle.dump(hindpawPos, open(self.analysisLocation + '%s_%s_%s_hindpawLocations.p' % (mouse, date, rec), 'wb'))
        pickle.dump(rungs, open( self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec), 'wb' ) )


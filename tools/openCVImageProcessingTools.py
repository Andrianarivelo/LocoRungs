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

        # tracking parameters #########################
        self.thresholdValue = 0.8 # in %
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

        Radius = 1400
        xCenter = 1485
        yCenter = 1545

        ########################################################################
        # loop to find correct wheel mask
        nIt = 0
        while True:
            wheelMask = np.zeros((self.Vheight, self.Vwidth))
            wheelMaskInv = np.zeros((self.Vheight, self.Vwidth))

            mask = np.sqrt((imgGrid[0] - xCenter) ** 2 + (imgGrid[1] - yCenter) ** 2) > Radius
            maskInv = np.sqrt((imgGrid[0] - xCenter) ** 2 + (imgGrid[1] - yCenter) ** 2) < Radius
            wheelMask[mask] = 1
            wheelMaskInv[maskInv] = 1
            wheelMask = np.array(wheelMask, dtype=np.uint8)
            wheelMaskInv = np.array(wheelMaskInv, dtype=np.uint8)

            imgMaskInv = cv2.bitwise_and(img, img, mask=wheelMaskInv)
            imgMask = cv2.bitwise_and(img, img, mask=wheelMask)
            cv2.imshow("Wheel mask %s" % nIt , imgMask)
            cv2.waitKey(1000)
            print 'current radius, xCenter, yCenter : ' , Radius, xCenter, yCenter
            var = raw_input("Enter new radius, xCenter, yCenter coordinates (separated by commas), otherwise press 'k' : ")
            #print "you entered", var
            if var == 'k':
                break
            else:
                (Radius, xCenter, yCenter) = map(int, var.split(','))
                nIt += 1
                if nIt >= 2:
                    cv2.destroyWindow("Wheel mask %s" % (nIt-2))

        print 'masking after loop, Radius = %s, xCenter %s, yCenter = %s' % (Radius, xCenter, yCenter)

        # print wheelMask
        maxStep = 40
        maxStepCurrent = maxStep
        nF = 1
        rungs = []
        pawPos = []
        # pawPos.append([-2,0,1,np.pi*bbox[2]*bbox[3]/4.])
        pawPos.append([0, 'f', [], bboxFront[:2], np.pi * bboxFront[2] * bboxFront[3] / 4.])
        pawPos.append([0, 'h', [], bboxHind[:2], np.pi * bboxHind[2] * bboxHind[3] / 4.])
        #hindPawPos.append([0, [], bboxHind[:2], np.pi * bboxHind[2] * bboxHind[3] / 4.])
        checkPos = -1
        # loop over all images in video
        while True:
            # Read a new frame
            ok, img = self.video.read()
            if not ok:
                break
            orig = img.copy()
            orig2 = img.copy()
            orig3 = img.copy()
            # while (1):
            # ret, frame = cap.read()

            # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgInv = cv2.bitwise_and(img, img, mask=wheelMaskInv)
            img = cv2.bitwise_and(img, img, mask=wheelMask)

            imgGWheel = cv2.cvtColor(imgInv, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(imgGWheel, cv2.HOUGH_GRADIENT, 1, 22, param1=50, param2=20, minRadius=30, maxRadius=40)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(imgInv, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv2.circle(imgInv, (i[0], i[1]), 2, (0, 0, 255), 3)
                    cv2.circle(orig3, (i[0], i[1]), 2, (0, 0, 255), 3)
                    cv2.line(orig3, (i[0], i[1]), (500, -20), (255, 0, 0), 3)
                    rungs.append([nF, i[0], i[1], 500, -20])
                if self.showImages:
                    cv2.imshow('detected circles', imgInv)

            if self.showImages:
                cv2.imshow("Masking", imgInv)
            k = cv2.waitKey(10) & 0xff
            if k == 27: break

            # blur image and apply threshold
            imgGMouse = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(imgGMouse, (5, 5), 0)
            minMaxL = cv2.minMaxLoc(blur)
            ret, th1 = cv2.threshold(blur, minMaxL[1] * self.thresholdValue, 255, cv2.THRESH_BINARY)
            # print ret, th1
            # perform edge detection, then perform a dilation + erosion to
            # close gaps in between object edges
            edged = cv2.Canny(th1, 50, 100)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)

            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]

            # sort the contours from left-to-right and initialize the
            # 'pixels per metric' calibration variable
            (cnts, _) = contours.sort_contours(cnts)
            pixelsPerMetric = None
            # print cnts

            nCnts = 0
            cntDistances = []
            cntArea = []
            rois = []
            for c in cnts:
                # if the contour is not sufficiently large, ignore it
                if cv2.contourArea(c) < self.minContourArea:
                    continue
                # print 'contourArea : ',cv2.contourArea(c)
                # compute the rotated bounding box of the contour
                ell = cv2.fitEllipse(c)
                # print ell
                cntDistances.append(dist.euclidean(pawPos[checkPos][3], ell[0]))
                cntArea.append(np.pi * (ell[1][0] / 2.) * (ell[1][1] / 2.0))
                rois.append(ell)
                orig = cv2.ellipse(orig, ell, (255, 0, 0), 2)
                nCnts += 1
            # find ellipse which is the best continuation of the previous ones
            # print 'nContours, Dist, Areaa : ', nCnts, cntDistances, cntArea
            print nF,
            if nCnts > 0:
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
                Achange = abs(np.asarray(cntArea) / pawPos[checkPos][4] - 1.) * 100.  # in percent
                DWeight = 0.5
                print checkPos, Dchange, Achange,
                Pindex = np.argmin(DWeight * Dchange + (1. - DWeight) * Achange)
                # Dindex = np.argmin(abs(np.asarray(cntDistances)-Dprojection))
                # print Dindex, Dprojection, pawPos[-2:]
                # if abs(cntArea[Dindex]/pawPos[-1][3] - 1.) < 0.2:
                # print cntDistances[Aindex]
                if (nCnts == 1) or (Dchange[Pindex] < abs(checkPos) * 60.) and (Achange[Pindex] < abs(checkPos) * 200.):
                    print 'success', Pindex, Dchange[Pindex], Achange[Pindex]
                    pawPos.append([nF,'f', rois[Pindex], rois[Pindex][0], cntArea[Pindex]])
                    orig = cv2.ellipse(orig, rois[Pindex], (0, 255, 0), 2)
                    orig2 = cv2.ellipse(orig2, rois[Pindex], (0, 255, 0), 2)
                    orig3 = cv2.ellipse(orig3, rois[Pindex], (0, 255, 0), 2)
                    checkPos = -1  # pointLoc = rois[Dindex][0]  # maxStepCurrent = maxStep
                else:
                    print 'failure'
                    pawPos.append([nF,'', -1, -1, -1])
                    checkPos -= 1
            else:
                print 'failure'
                pawPos.append([nF,'', -1, -1, -1])
                checkPos -= 1
            if self.showImages:
                cv2.imshow("Image", orig)

            # k = cv2.waitKey(1) & 0xff
            if k == 27: break
            nF += 1

        cv2.destroyAllWindows()

        # save tracked data
        #(test,grpHandle) = self.h5pyTools.getH5GroupName(self.f,'')
        #self.h5pyTools.createOverwriteDS(grpHandle,'angularSpeed',angularSpeed,['monitor',monitor])
        #self.h5pyTools.createOverwriteDS(grpHandle,'linearSpeed', linearSpeed)
        #self.h5pyTools.createOverwriteDS(grpHandle,'walkingTimes', wTimes, ['startTime',startTime])

        pickle.dump(pawPos, open( self.analysisLocation + '%s_%s_%s_pawLocations.p' % (mouse, date, rec), 'wb' ) )
        pickle.dump(rungs, open( self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec), 'wb' ) )


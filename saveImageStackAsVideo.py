import numpy as np
import cv2

cap = cv2.VideoCapture('chaplin.mov')

w = 480
h = 640

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX') #(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20., (h,w))

ret = True
while(cap.isOpened()):
    frameRaw = np.random.rand(w,h)*255.

    frame2 = np.array(frameRaw,dtype=np.uint8)
    #ret, frame = cap.read()
    frame = cv2.cvtColor(frame2, cv2.COLOR_GRAY2RGB)
    if ret == True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
import cv2 as cv
import numpy as np
import time

import hand_tracking as ht
import math
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


ptime=0
ctime=0
wcam,hcam=640,480

cap=cv.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,wcam)



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()

volrange=volume.GetVolumeRange()
minvol=volrange[0]
maxvol=volrange[1]
#volume.SetMasterVolumeLevel(-20.0, None)



detector=ht.HandDetector(detectionCon=0.7)
while True:
    success, img = cap.read()
    img=detector.findHands(img)
    lmlist= detector.findPosition(img,draw=False)
    if len(lmlist)!=0:  
        #print(lmlist[4],lmlist[8])
        x1,y1=lmlist[4][1], lmlist[4][2]
        x2,y2=lmlist[8][1], lmlist[8][2]
        cx,cy= ((x1+x2)//2), ((y1+y2)//2)

        cv.circle(img,(x1,y1),10,(255,0,0),cv.FILLED)
        cv.circle(img,(x2,y2),10,(255,0,0),cv.FILLED)
        cv.circle(img,(cx,cy),10,(255,0,0),cv.FILLED)
        cv.line(img,(x1,y1),(x2,y2),(255,255,0),3)
        length=math.hypot(x2-x1, y2-y1)
        #print(length)
        vol=np.interp(length,[50,300],[minvol,maxvol])
        volume.SetMasterVolumeLevel(vol, None)
        print(vol)

        if length<50:
            cv.circle(img,(cx,cy),10,(255,0,255),cv.FILLED)

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime 
    cv.putText(img, str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX, 3, (255,255,0),3)

    cv.imshow("img",img)


    key=cv.waitKey(1)

    if key%256 == 27:
            break


cap.release()
cv.destroyAllWindows()
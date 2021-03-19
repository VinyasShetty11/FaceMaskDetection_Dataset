import winsound
import numpy as np
frequency= 2500    
duration =1000    #1000ms==1 sec
from newmodel import model as new_model
import cv2 #pip install opencv-python
path="haarcascade_frontalface_default.xml"
font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN
#set the rectangle background to white
rectangle_bgr=(255,255,255)
#set a black image
img=np.zeros((500,500))
#set some text
text="Some text in a box"
#get the width and height of the text box
(text_width,text_height)=cv2.getTextSize(text,font,fontScale=font_scale,thickness=1)[0]
#set the text satart position
text_offset_x=10
text_offset_y=img.shape[0] -25
#make the coords of the box with small padding of two pixels
box_coords=((text_offset_x,text_offset_y),(text_offset_x + text_width + 2,text_offset_y - text_height - 2))
cv2.rectangle(img,box_coords[0],box_coords[1],rectangle_bgr,cv2.FILLED)
cv2.putText(img,text,(text_offset_x,text_offset_y),font,fontScale=font_scale, color=(0,0,0),thickness=1)

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
#check if the webcam is opened correctly
if not cap.isOpened():
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
if not cap.isOpened():
    raise IOError("cannot open Webcam")

while True:
    ret,frame=cap.read()
    #eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascades_eye.xml')
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    faces=faceCascade.detectMultiScale(gray,1.1,4)

    for x,y,w,h in faces:
        roi_gray =gray[y:y+h,x:x+w]
        roi_color =frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        faces=faceCascade.detectMultiScale(roi_gray)
        if len(faces)==0:
            print("Face Not Detected")
        else:
            for(ex,ey,ew,eh) in faces:
                face_roi=roi_color[ey:ey+eh,ex:ex +ew]
    final_image=cv2.resize(face_roi,(224,224))
    final_image= np.expand_dims(final_image,axis=0)   #need fourth dimension
    final_image=final_image/255.0
    font=cv2.FONT_HERSHEY_SIMPLEX
    Predicitions=new_model.predict(final_image)

    font_scale=1.5
    font=cv2.FONT_HERSHEY_PLAIN

    if(Predicitions>0):
        status="NoMask"

        x1,y1,w1,h1=0,0,175,75
        #draw black background rectangle
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        #add text
        cv2.putText(frame,status,(x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,(0,255,0),2)

        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    else:
        status="Face Mask"

        x1,y1,w1,h1 =0,0,175,75
        #draw black background rectangle
        cv2.rectangle(frame,(x1,x1),(x1 + w1,y1+ h1),(0,0,0),-1)
        #add text
        cv2.putText(frame,status,(x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # #print(faceCascade.empty())
        # #faces=faceCascade.detecMultiScale(gray,1.1,4)

        # #Draw a rectangle around the faces
        # #for(x,y,w,h) in faces:
        # #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        # #USe putText()method for
        # #inserting text on video
    cv2.imshow('Face MAsk Detection ',frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destoryALlWindows()



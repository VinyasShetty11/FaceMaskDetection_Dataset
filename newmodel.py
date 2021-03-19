import tensorflow as tf 
import cv2      #pip install opencv
import os 
import random
import pickle
import matplotlib.pyplot as plt #pip install matplotlib
import numpy as np # pip install numpy
from tensorflow import keras
from tensorflow.keras import layers

img_array=cv2.imread("Dataset/Face_Mask/00000_MAsk.jpg")
plt.imshow(img_array)

plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
img_array.shape

Datadirectory="Dataset/" #training dataset
Classes=["Face_Mask","No_Mask"]
for category in Classes:
    path = os.path.join(Datadirectory,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break
img_size=224
new_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array,cv2.COLOR_BGR2RGB))
plt.show()

#######reading the image and converting all to array
training_Data=[]
def create_training_Data():
    for category in Classes:
        path=os.path.join(Datadirectory,category)
        class_num=Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_Data()
print(len(training_Data))

random.shuffle(training_Data)
x=[]    #data
y=[] #label
for features,label in training_Data:
    x.append(features)
    y.append(label)

X=np.array(x).reshape(-1,img_size,img_size,3)
X.shape

#normalize the data
X=X/255.0
y[1000]

Y=np.array(y)

pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in=open("X.pickle","rb")
X=pickle.load(pickle_in)

pickle_in=open("y.pickle","rb")
y=pickle.load(pickle_in)


model=tf.keras.applications.mobilenet.MobileNet() #pre-trained model
model.summary()
base_input =model.layers[0].input
base_output =model.layers[-4].output
Flat_layer= layers.Flatten()(base_output)
final_output = layers.Dense(1)(Flat_layer)
final_output=layers.Activation('sigmoid')(final_output)
new_model= keras.Model(base_input,outputs=final_output)
new_model.summary()

#settings for binary classification(Face mak/with out mask)
new_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
new_model.fit(X,Y,epochs =1,validation_split=0.1)
new_model.save('my_mode13.h5')
new_model=tf.keras.models.load_model('my_mode13.h5')

#Checking the network for predictions
frame=cv2.imread('Dataset/Face_Mask/00002_Mask.jpg')
plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
final_image=cv2.resize(frame,(224,224))
final_image=np.expand_dims(final_image,axis=0)      #need fourth dimension
final_image=final_image/255.0
Predictions=new_model.predict(final_image)
Predictions

#checking the network for unknown imagws
frame=cv2.imread('sad women.jpg')
frame.shape
plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
facesCascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
gray.shape

faces=facesCascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces:
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=frame[y:y+h,x:x+w]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    faces=facesCascade.detectMultiScale(roi_gray)
    if len(faces)==0:
        print("Face not detected")
    else:
        for(ex,ey,ew,eh) in faces:
            face_roi=roi_color[ey:ey+eh,ex:ex +ew]
plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
plt.imshow(cv2.cvtColor(face_roi,cv2.COLOR_BGR2RGB))
final_image=cv2.resize(face_roi,(224,224))
final_image=np.expand_dims(final_image,axis=0)      ##need fourth dimwnsion
final_image=final_image/255.0

Predictions=new_model.predict(final_image)
Predictions


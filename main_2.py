import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from keras.utils import to_categorical,plot_model
from keras.callbacks import CSVLogger
import pylab as pl
import os
from skimage.transform import resize,rotate
import numpy as np
import pdb

def makemodel():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model

##  load realBags img
rootpath = './RealBags/'
imgdir = os.listdir(rootpath)
clsInfo = []
data = []
label = []
ims = []
rescaleSize = (100,100)
for clsname in imgdir:
    if os.path.isdir(os.path.join(rootpath,clsname)):
        clsInfo.append(clsname)
        for imname in os.listdir(os.path.join(rootpath,clsname)):
            im0 = pl.imread(os.path.join(rootpath,clsname,imname))
            im = resize(im0,rescaleSize,mode='reflect')
            data.append(np.resize(im,(1,)+rescaleSize+(3,)))
            label.append(1)
            ims.append(im0)
            for arg in np.arange(-45,45,10):
                im = resize(rotate(im0,arg,resize=True,cval=1),rescaleSize,mode='reflect')
                data.append(np.resize(im,(1,)+rescaleSize+(3,)))
                label.append(1)

## load Counterfeits bags img
rootpath = './Counterfeits/'
imgdir = os.listdir(rootpath)
rescaleSize = (100,100)
for imname in imgdir:
    if imname[-3:] != 'jpg' and imname[-3:] != 'png':
        continue
    im0 = pl.imread(os.path.join(rootpath,imname))
    im = resize(im0,rescaleSize,mode='reflect')
    data.append(np.resize(im,(1,)+rescaleSize+(3,)))
    label.append(0)
    ims.append(im0)
    for arg in np.arange(-45,45,10):
        im = resize(rotate(im0,arg,resize=True,cval=1),rescaleSize,mode='reflect')
        data.append(np.resize(im,(1,)+rescaleSize+(3,)))
        label.append(0)


## split train data and test data
id = np.arange(len(label))
np.random.shuffle(id)

num = int(0.8*len(label))
trainData = [data[ii] for ii in id[0:num]]
trainLabel = [label[ii] for ii in id[0:num]]
testData = [data[ii] for ii in id[num:]]
testLabel = [label[ii] for ii in id[num:]]

trainData=np.concatenate(trainData,axis=0)
testData = np.concatenate(testData,axis=0)
trainY = to_categorical(trainLabel,2)
testY = to_categorical(testLabel,2)


model = makemodel()
plot_model(model,to_file='model_part2.png',show_shapes=True)
csv_logger = CSVLogger('training_part2.log')
model.fit(trainData,trainY,batch_size=64,epochs=100,verbose=1,validation_data=(testData,testY),callbacks=[csv_logger])
predictY = model.predict(testData)

acc = np.sum(np.array(testLabel) == np.argmax(predictY,axis=1))/float(len(predictY))
print('accurate : %f%%'%(acc*100))
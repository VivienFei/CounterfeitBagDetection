import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.utils import to_categorical,plot_model
from keras.callbacks import CSVLogger
import pylab as pl
import os
from skimage.transform import resize,rotate
import numpy as np
import pdb

def makemodel(nCls):
    model = Sequential()
    # filter size: 7*7，numbers:16，input image size: 100*100*3
    # filter大小7*7，数量32个，原始图像大小3,150,150
    model.add(Conv2D(16, (7, 7), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nCls, activation='softmax')) #  matt,几个分类就要有几个dense
    model.compile(loss='categorical_crossentropy', optimizer='adadelta') # Configures the model for training.
    return model

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
        clsId = len(clsInfo)-1
        for imname in os.listdir(os.path.join(rootpath,clsname)):
            im0 = pl.imread(os.path.join(rootpath,clsname,imname))
            im = resize(im0,rescaleSize,mode='reflect')
            data.append(np.resize(im,(1,)+rescaleSize+(3,)))
            label.append(clsId)
            ims.append(im0)
            for arg in np.arange(-45,45,10):
                im = resize(rotate(im0,arg,resize=True,cval=1),rescaleSize,mode='reflect')
                data.append(np.resize(im,(1,)+rescaleSize+(3,)))
                label.append(clsId)

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
trainY = to_categorical(trainLabel,12)
testY = to_categorical(testLabel,12)


nCls = 12
model = makemodel(nCls)
#plot_model(model,to_file='model_part.png',show_shapes=True)
csv_logger = CSVLogger('training_part1.log')
model.fit(trainData,trainY,batch_size=64,epochs=100,validation_data=(testData,testY),callbacks=[csv_logger])
predictY = model.predict(testData)

acc = np.sum(np.array(testLabel) == np.argmax(predictY,axis=1))/float(len(predictY))
print('accurate : %f%%'%(acc*100))

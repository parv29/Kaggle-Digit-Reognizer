import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import random
test=pd.read_csv('test.csv')
test=test/255.0
tet=test.iloc[:,0:].values
data=pd.read_csv('train.csv')
training=data.iloc[:,1:].values/255.0
testing=data.iloc[:,0].values
X_train=data.iloc[:,1:].values/255.0
y_train=data.iloc[:,0].values
i = random.randint(1,42000) 
plt.imshow( training[i,0:].reshape((28,28)) ) 

plt.imshow( training[i,0:].reshape((28,28)) , cmap = 'gray') 

label = training[i,0]

from sklearn.model_selection import train_test_split

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.1, random_state = 12345)

X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))
tet=tet.reshape(tet.shape[0], *(28, 28, 1))

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

cnn_model = Sequential()

cnn_model.add(Conv2D(64,3, 3, input_shape = (28,28,1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Dropout(0.25))
cnn_model.add(Conv2D(64,3, 3,  activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Dropout(0.2))


cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim = 1280, activation = 'relu'))
cnn_model.add(Dropout(0.1))

cnn_model.add(Dense(output_dim = 1280, activation = 'relu'))
cnn_model.add(Dropout(0.2))



cnn_model.add(Dense(output_dim = 10, activation = 'sigmoid'))
cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])


history = cnn_model.fit(X_train,
                        y_train,
                        batch_size = 512,                
                        nb_epoch = 75,
                        verbose = 1,
                        validation_data = (X_validate, y_validate))

sample=pd.read_csv('sample_submission.csv')
imageid=sample["ImageId"]
imageid=imageid.T
column=cnn_model.predict_classes(tet)
model=pd.DataFrame(pd.DataFrame({"ImageId":imageid,"Label":column}))
model.to_csv("mni.csv",index=False)

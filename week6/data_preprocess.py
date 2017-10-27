from scipy.io import loadmat
import numpy as np

#load data
data = loadmat('concrete_data.mat')['A']

#add error to data
data[:,1] += np.random.normal(0,10,data.shape[0])
np.savetxt('error_data.txt',data)

#data pre-processing
max_strain = np.max(data[:,0])
max_stress = np.max(data[:,1])
min_strain = np.min(data[:,0])
min_stress = np.min(data[:,1])

#cannot use standard normal distribution here: will bias on mean value
data[:,0] = ((data[:,0] - min_strain)/(max_strain - min_strain))*2-1
data[:,1] = ((data[:,1] - min_stress)/(max_stress - min_stress))*2-1

#shuffle data, choose 75% as training and 25% for validation
np.random.shuffle(data)
train = data[0:int(data.shape[0]*0.75) , ]
test = data[int(data.shape[0]*0.75): , ]

x_train = train[:,0]
y_train = train[:,1]

x_test = test[:,0]
y_test = test[:,1]

#build structure
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(10, activation='selu', input_dim=1))
#model.add(Dropout(0.5))
model.add(Dense(10, activation='selu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='selu'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=30, epochs=50,
          validation_data=(x_test, y_test))

model.summary()
print(history.history.keys())
# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

result = []
for i in range(380):
    input = ((i*0.00001 - min_strain)/(max_strain - min_strain))*2-1
    output = model.predict(np.asarray([input]))
    output = (output+1)/2*(max_stress - min_stress)+min_stress
    result.append([i*0.00001,output])

result = np.asarray(result)
np.savetxt('Simple_increment.txt', result)
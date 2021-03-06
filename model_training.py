from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense 
import pickle 

features = pickle.load(open('features.pkl', 'rb')) 
labels = pickle.load(open('labels.pkl', 'rb'))

features = features.reshape(-1, 250, 250, 1)
#features = features / 255

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape = features[1:], activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(features, labels, epochs=10, validation_split=0.1)

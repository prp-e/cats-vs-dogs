from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense 
import pickle 

features = pickle.load(open('features.pkl', 'rb')) 
labels = pickle.load(open('labels.pkl', 'rb'))

model = Sequential()


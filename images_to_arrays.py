import numpy as np 
import os 
import cv2 
import random
import pickle

DIRECTORY = r'dogscats/train'
CATEGORIES = ['cats', 'dogs']
IMAGE_SIZE = 250

data = []

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for image in os.listdir(folder):
        image_path = os.path.join(folder, image)
        print(f'Working on {image_path}')
        image_arr = cv2.imread(image_path)
        image_arr = cv2.resize(image_arr, (IMAGE_SIZE, IMAGE_SIZE))
        data.append([image_arr, label])

random.shuffle(data)
x = []
y = []

for features, labels in data: 
    x.append(features)
    y.append(labels)

print("Pre-process is done successfully.")

x = np.array(x)
y = np.array(y)

pickle.dump(x, open('features.pkl', 'wb'))
pickle.dump(y, open('labels.pkl', 'wb'))

print("Saved to files!")
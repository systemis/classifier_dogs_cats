from os import listdir
from numpy import save
from numpy import load 
from matplotlib.image import imread
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from keras_preprocessing.image import load_img, img_to_array
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2;


data_filenames = {
  'photos': 'dogs_vs_cats_photos.txt',
  'labels': 'dogs_vs_cats_labels.txt'
}

folder = 'data_sets/train/'
test_folder = 'data_sets/test/'
testimage = folder + 'cat.1.jpg'
train_x_all = [];

def test_show_dog():
  for i in range(9):
    plt.subplot(300 + 1 + i)
    filename = folder + 'dog.' + str(i) + '.jpg'
    image = imread(filename)
    plt.imshow(image)

  plt.show();

# Save ndarray 
def saving_data():
  photos, labels = list(), list()
  for filename in listdir(folder):
    label = 0.0
    image = img_to_array(load_img(folder + filename, target_size=(200, 200)))# Read image 
    image = reduce_dimension(image).tolist() # Dimensionly reducion 
    if filename.startswith('cat'): # label addition 
      label = 1.0

    photos.append(image)
    labels.append(label)

  # Write file 
  photosFile = open(data_filenames['photos'], 'w')
  photosFile.write(json.dumps(photos))
  photosFile.close()
  labelsFile = open(data_filenames['labels'], 'w')
  labelsFile.write(json.dumps(labels))
  labelsFile.close()

  print('Saved data train !! ')

# Dimensionly reduction 
def reduce_dimension(image):
  scale_percent = 8
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
  image = image.reshape(image.shape[0] * image.shape[1] * image.shape[-1])

  return image.astype(int);
  
# Load training data 
def load_training_data():
  pFile = open(data_filenames['photos'])
  photos = np.array(json.loads(pFile.read()))
  pFile.close()

  lFile = open(data_filenames['labels'])
  labels = np.array(json.loads(lFile.read()))
  lFile.close()
  return (photos, labels)

# Checking get image and reducing dimensions
def testing():
  imgs = list() 
  for i in range(10): 
    image = img_to_array(load_img(folder + 'cat.' + str(i) + '.jpg', target_size=(200, 200)))
    image = reduce_dimension(image).tolist();
    imgs.append(image)

  file = open('test.txt', 'w')
  file.write(json.dumps(imgs))
  file.close()

# saving_data();
# image = img_to_array(load_img(testimage))
# image = reduce_dimension(image)
# print image.shape 
X_train, y_train = load_training_data();
X_test, y_test = X_train[1000:2000], y_train[1000:2000]

loged = linear_model.LogisticRegression(C=1e5)
loged.fit(X_train, y_train)

y_pred=loged.predict(X_test)
print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred.tolist())))

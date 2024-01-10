import os

for directory in os.listdir('train/'):
  image_list = os.listdir('train/' + directory)
  if len(image_list) == 0:
    os.rmdir('train/' + directory) # if not empty throws exception

import os
import shutil
import random

for directory in os.listdir('train/'):
  image_list = os.listdir('train/' + directory)
  if len(image_list) >= 5:
    if not os.path.exists('val/' + directory):
      os.makedirs('val/' + directory, exist_ok=True)
    # Random sample 20% of the train images and move it to validation
    random.shuffle(image_list)
    for img in image_list[0: int(0.2*len(image_list))]:
      src = 'train/' + directory +'/' + img
      destination = 'val/' + directory
      shutil.move(src, destination)
    



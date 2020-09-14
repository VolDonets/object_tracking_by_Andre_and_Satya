import dlib
import glob
import cv2
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pyautogui as pyg
import shutil

from sklearn_porter import Porter

directory = 'train_images_h'
box_file = 'boxes_h.txt'
data = {}
image_indexes = [int(img_name.split('.')[0]) for img_name in os.listdir(directory)]
np.random.shuffle(image_indexes)
f = open(box_file, "r")
box_content = f.read()
box_dict = eval('{' + box_content + '}')
f.close()

for index in image_indexes:
    img = cv2.imread(os.path.join(directory, str(index) + '.png'))
    bounding_box = box_dict[index]
    x1, y1, x2, y2 = bounding_box
    dlib_box = [dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)]
    data[index] = (img, dlib_box)


no_of_samples = 10
image_names = os.listdir(directory)
np.random.shuffle(data)
cols = 5
rows = int(np.ceil(no_of_samples / cols))
plt.figure(figsize=(cols * cols, rows * cols))

for i in range(no_of_samples):
    d_box = data[i][1][0]
    left, top, right, bottom = d_box.left(), d_box.top(), d_box.right(), d_box.bottom()
    image = data[i][0]
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)
    plt.subplot(rows, cols, i + 1);
    plt.imshow(image[:, :, ::-1]);
    plt.axis('off');


percent = 0.8
split = int(len(data) * percent)

images = [tuple_value[0] for tuple_value in data.values()]
bounding_boxes = [tuple_value[1] for tuple_value in data.values()]

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = False
options.C = 8
st = time.time()
#detector = dlib.train_simple_object_detector(images[:split], bounding_boxes[:split], options)
detector = dlib.train_simple_object_detector(images, bounding_boxes, options)
print('Training Completed, Total Time taken: {:.2f} seconds'.format(time.time() - st))

#file_name = 'models/Hand_Detector_v8_c8.svm'
#detector.save(file_name)


win_det = dlib.image_window()
win_det.set_image(detector)

print("Training Metrics: {}".format(dlib.test_simple_object_detector(images[:split], bounding_boxes[:split], detector)))
print("Testing Metrics: {}".format(dlib.test_simple_object_detector(images[split:], bounding_boxes[split:], detector)))

porter = Porter(detector, language='C')
output = porter.export(embed_data=True)
print(output)
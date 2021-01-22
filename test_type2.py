import tensorflow as tf
import os
import re
import numpy

from PIL import Image

import data_preprocess as d
images_path = "./archive/single_test/"
img_size = (255,255)
model = tf.keras.models.load_model("./16x3-CNN-l_mod3.model")

model.summary()
# ADD IMAGE NAME IN IMAGE WITH EXTENSION#####
image = 'PLACEHOLDER'
img = Image.open(os.path.join(images_path, image))
face = img.resize(img_size).convert("RGB")
face.show()
face_img = numpy.asarray(face).reshape(-1, img_size[0], img_size[1], 3)
face_img = face_img / 255

prediction = model.predict([face_img])
print(prediction)
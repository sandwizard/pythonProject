import tensorflow as tf
import os
import re
import numpy

from PIL import Image

import data_preprocess as d
images_path = "./archive/batch_test/"
labels_path = "./archive/annotations/"
images = list(sorted(os.listdir(images_path)))
labels = list(sorted(os.listdir(labels_path)))
img_size=(255,255)
model = tf.keras.models.load_model("./16x3-CNN-l_mod3.model")

for image in list(sorted(os.listdir(images_path))):
    annotation = re.sub('png', 'xml', image)
    #print('image:' + image)
    #print('label:' + annotation)
    targt = d.generate_target(image, os.path.join(labels_path, annotation))
    img = Image.open(os.path.join(images_path, image))
    faces = []
    labels=[]
    for box in targt['boxes']:
        #xmin, ymin, width, height
        face = img.crop((box[0], box[1], box[2], box[3]))

        face = face.resize(img_size).convert("RGB")

        face_img = numpy.array(face).reshape(-1, img_size[0], img_size[1], 3)
        face.show()
        face_img = face_img / 255

        prediction = model.predict([face_img])
        print(prediction)






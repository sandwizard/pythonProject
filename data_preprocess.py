from PIL import Image
import os
import re
import cv2
import numpy
import random
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelBinarizer
import pickle


import tensorflow as tf


# prep the data
# to do import the data used in kaggle
# get the anotations to get pixel specific data
# convert data to gray scale
# rescale data to a uniform res
# convert to matrix and store with labels
images_path = "./archive/images/"
labels_path = "./archive/annotations/"


extra_without = "./archive/extra/without_mask/"
# list of all images and labels sorted order


###################################################################################


# generate a box over the face in image obj is the annotation of the image
def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


# generate a label showing the correct class 0- no mask 1-with mask 2- incorrectly worn
def generate_label(obj):

    if obj.find('name').text == "with_mask":
        y = 0
        return y
    elif obj.find('name').text == "mask_weared_incorrect":
        y = 2
        return y
    elif obj.find('name').text=="without_mask":
        y = 1
        return y


# pass the image and xml file ang get the required target data using above functions
def generate_target(image_id, file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        num_objs = len(objects)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in objects:
            if(generate_label(i)!=2):
                boxes.append(generate_box(i))
                labels.append(generate_label(i))

        # need to convert labels to tensor for easier no need to convert others since not used in calculation

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        return target




# run once and save trainning daata

def get_trainning_set(img_size):
    trainning_data = []

    # itterate through all images
    for image in list(sorted(os.listdir(images_path))):

        annotation = re.sub('png', 'xml', image)
        #print('image:' + image)
        #print('label:' + annotation)
        target = generate_target(image, os.path.join(labels_path, annotation))
        img = Image.open(os.path.join(images_path, image))
        img = img.convert("RGB")



        for box in target['boxes']:
            face = img.crop((box[0], box[1], box[2], box[3]))
            face = face.resize(img_size)






            label = target['labels'][target['boxes'].index(box)]

            trainning_data.append([numpy.asarray(face), label])

################################ remove later only for eas of testing ##############################
                #faces.append([numpy.array(face), label])


        # test if only target shows

            #cv2.imshow('main', im)
################## use faces instead of trainning ata to only see faces of one ig at a time   # remove later#######
                # for veiwing images
            ''' 
            while True:
                for face in faces:
                    cv2.imshow(str(face[1])+' :lable '+str(face[0])[:9], face[0])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
                    '''





    return trainning_data

def create_trainning_set(img_size):
    trainning_data = get_trainning_set(img_size)
    random.shuffle(trainning_data)
    X=[]
    y=[]

    for features,label in trainning_data:
        X.append(features)
        y.append(label)
    X = numpy.array(X).reshape(-1, img_size[0],img_size[1],3)

    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)

    pickle_out = open('X_'+str(img_size[0])+'_'+str(img_size[1])+'.pickle','wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open('y_' + str(img_size[0]) + '_' + str(img_size[1]) + '.pickle', 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()
    return
create_trainning_set((255,255))
CATEGORIES = ["without_mask","with_mask", "mask_weared_incorrect"]  # will use this to convert prediction num to string value



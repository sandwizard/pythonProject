from PIL import Image

import cv2
import numpy
import tensorflow as tf

import os
cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
# detector is trained mtcnn model used to detect a face

dir_path = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists('Output'):
  os.makedirs('Output')

model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')






cam = cv2.VideoCapture(0)
img_size=(255, 255)
CATEGORIES = ["without_mask","with_mask","mask_weared_incorrect"]
# a cam to detect faces
#and take that img and use on neural network for mask
model_mask = tf.keras.models.load_model("./16x3-CNN-l_mod3.model")
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (600, 600))
    (h, w) = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()
    # faces = faceCascade.detectMultiScale(blob,
    #                                      scaleFactor=1.1,
    #                                      minNeighbors=5,
    #                                      minSize=(60, 60),
    #                                      flags=cv2.CASCADE_SCALE_IMAGE)
    locs = []
    count = 0
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]
# if the algorithm is more than 16.5% confident that the      detection is a face, show a rectangle around it
        if (confidence > 0.365):


            count = count + 1

            face = frame[startY:endY, startX:endX]
            cv2.imwrite("color_img.jpg",face)
            cv2.imshow("color_img.jpg",3)
            face = Image.open("color_img.jpg")
            face = face.resize(img_size).convert("RGB")

            face_img = numpy.array(face).reshape(-1, img_size[0], img_size[1], 3)

            face_img = face_img / 255
            prediction = model_mask.predict([face_img])


            p = numpy.where(prediction[0] == prediction[0].max())
            print(prediction)
            print(prediction)
            if prediction>0.7:
                print("without_mask")
                cv2.putText(frame, 'without_mask', (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)


            else:
                print( "with_mask")
                cv2.putText(frame, 'with_mask', (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)



            # Display the resulting frame


    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

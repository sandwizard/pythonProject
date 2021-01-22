from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
import tensorflow as tf
from PIL import Image
import numpy

# load the model
model = tf.keras.models.load_model("./16x3-CNN-l_mod3.model")
images_path = "./archive/single_test/"
img_size = (255, 255)

# redefine model to output right after the first hidden layer


# load the image with the required shape
img = Image.open('./archive/single_test/aa.jpg')

# convert the image to an array
# expand dimensions so that it represents a single 'sample'
img= img.resize(img_size).convert("RGB")
# prepare the image (e.g. scale pixel values for the vgg)

# get feature map for first hidden layer
ixs = [0,3,6]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
model.summary()
# convert the image to an array

# expand dimensions so that it represents a single 'sample'

# prepare the image (e.g. scale pixel values for the vgg)
img = numpy.array(img).reshape(-1, img_size[0], img_size[1], 3)
img / 255
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot the output from each block
square = 2
for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(fmap[0, :, :, ix-1],cmap="gray")
			ix += 1
	# show the figure
	pyplot.show()
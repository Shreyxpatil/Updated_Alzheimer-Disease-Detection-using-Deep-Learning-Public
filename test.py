import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('ALZHEIMER.h5')

image = cv2.imread(r'Dataset\ModerateDemented\moderate.jpg')

img = Image.fromarray(image)

img = img.resize((64, 64))

img = np.array(img)

img = img / 255.0  

img = np.expand_dims(img, axis=0)

predictions = model.predict(img)

predicted_class = np.argmax(predictions)

print(f"Predicted class: {predicted_class}")

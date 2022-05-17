from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os, os.path


# Load the model
model = load_model('keras_model.h5')
imgs=[]
path='/dogs'
results=[0,0,0,0,0,0]
valid_images=['.jpg', '.gif', '.png', '.tga']
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image=imgs.append(Image.open(os.path.join(path,f)))
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
#####image = Image.open('husky2238.jpg')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
######image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
#image_array = np.asarray(image)
# Normalize the image
#####normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)

Pcls1=prediction[0,0]
Pcls2=prediction[0,1]
Pcls3=prediction[0,2]
Pcls4=prediction[0,3]
Pcls5=prediction[0,4]
Pcls6=prediction[0,5]

Predictions=[Pcls1, Pcls2, Pcls3, Pcls4, Pcls5, Pcls6]
indexOfClass=Predictions.index(max(Predictions))
if indexOfClass == 0:
    results[0]=results[0]+1
elif indexOfClass ==1:
    results[1]=results[1]+1
elif indexOfClass == 2:
    results[2]=results[2]+1
    #print('doing Something')
elif indexOfClass == 3:
    results[3]=results[3]+1
elif indexOfClass==4:
    results[4]=results[4]+1
elif indexOfClass==5:
    results[5]=results[5]+1
##################Debugging Code 


print(results)
# print(prediction)
# print(indexOfClass)
#print("result:")
#############################################


#print(prediction)

# Import necessary libraries
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('my_model.h5')  # Replace 'my_model.h5' with the path to your saved model

# Load and preprocess the new image
img_path = 'test/pet3.jpg'  # Replace 'your_image.jpg' with the name of your image
img = image.load_img(img_path, target_size=(300, 300))  # Load the image and resize it
img_array = image.img_to_array(img)  # Convert the image to a numpy array
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
img_array /= 255.0  # Rescale the pixel values to [0, 1]

# Predict the class of the new image
prediction = model.predict(img_array)

# Output the prediction
if prediction[0][0] > 0.5:
    print("The image is a dog.")
    # print(prediction)
   # print(img_array)
else:
    print("The image is a cat.")
    # print(prediction[0][0])
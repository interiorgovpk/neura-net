# Import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator

# Define the neural network architecture
model = Sequential([
    Flatten(input_shape=(300, 300, 3)),  # Flatten the input image of size 300x300 with 3 color channels (RGB)
    Dense(512, activation='relu'),       # First hidden layer with 512 neurons and ReLU activation
    Dense(1, activation='sigmoid')       # Output layer with 1 neuron and sigmoid activation for binary classification
])

# Compile the model with a binary crossentropy loss function and an optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create an instance of ImageDataGenerator for data augmentation (optional) and rescaling
datagen = ImageDataGenerator(rescale=1./255)

# Assuming you have a directory 'data/train' with subdirectories 'dogs' and 'cats'
# Each subdirectory should contain 10 images of the respective class
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(300, 300),  # Set the target size to match the image resolution
    batch_size=2,            # Set the batch size (you can adjust this based on your system's capabilities)
    class_mode='binary'      # Set the class mode to 'binary' for binary classification
)

# Train the model on the data
model.fit(train_generator, steps_per_epoch=5, epochs=10)  # Set steps_per_epoch to half the number of images to ensure each image is used once per epoch

# To predict on new images, you would use:
# predictions = model.predict(new_images)
model.save('my_model.h5')  # This will save your model to a file named 'my_model.h5'

# Note: 'new_images' should be preprocessed in the same way as the training images

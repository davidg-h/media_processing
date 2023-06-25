import os
from MLP import *
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import load_img

def image_loader(folder):
    image_list = []
    # Iterate over the files in the folder
    for filename in os.listdir(folder):
        if filename.endswith('.png'):  # Consider only PNG files, adjust the extension as needed
            image_path = os.path.join(folder, filename)
            img = load_img(image_path, target_size=(28, 28), color_mode='grayscale')
            # Normalize image
            (img, img_label) = normalize_img(img, filename)
            image_list.append(img)
    return image_list

def prediction(neural_network , cwd = os.path.dirname(__file__)):
    # Load the image
    folder = cwd + '/numbers'
    image_list = image_loader(folder)

    for img in image_list:
        # Expand dimensions to match the shape that the model expects
        plt.imshow(img, cmap='gray')
        plt.colorbar(label='Grayscale Value (0-255)')
        plt.show(block = False)
        
        img = np.expand_dims(img, axis=0)

        # Use the model to predict the digit
        predictions = neural_network.predict(img)
        for pred in predictions:
            print("Percentage of Number ", np.array(np.where(predictions[0] == pred)), ": ", pred)

        # The model gives probabilities for each digit, so take the digit with the highest probability
        predicted_digit = np.argmax(predictions[0])
        print('Predicted Number:', predicted_digit)
        
        user_input = input("Press Enter to continue to the next image or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        plt.close()
        

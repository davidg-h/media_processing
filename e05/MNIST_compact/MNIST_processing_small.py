import os
import tensorflow as tf

from MLP import create_mlp_model
from CNN import create_cnn_model
from PredictionAdapter import prediction
            
cwd = os.path.dirname(__file__)

""" # create MLP model and load it
create_mlp_model()
# retrain
create_mlp_model() """
mlp_number_guesser = tf.keras.models.load_model(cwd + '/mlp_number_guesser.h5')

""" # create CNN model
create_cnn_model()
# retrain
create_cnn_model() """
cnn_number_guesser = tf.keras.models.load_model(cwd + '/cnn_number_guesser.h5')

# Processing numbers/images with a model
user_input = input("Choose a neural network\n 1. Multi Layer Perceptron\n 2. Convolutional Neural Network\n or 'q' to quit: ")
while user_input != 'q':
    match user_input.lower():
        case '1':
            mlp_number_guesser.summary()
            prediction(mlp_number_guesser)
        case '2':
            cnn_number_guesser.summary()
            prediction(cnn_number_guesser)
    user_input = input("Choose a neural network\n 1. Multi Layer Perceptron\n 2. Convolutional Neural Network\n or 'q' to quit:")
    
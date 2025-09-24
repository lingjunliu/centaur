
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_per_image_standardization_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D image
    image1 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    input_dict1 = {"image": tf.constant(image1).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Image with negative values
    image2 = np.array([[[ -1, 2, -3], [4, -5, 6]], [[-7, 8, -9], [10, -11, 12]]], dtype=np.int32)
    input_dict2 = {"image": tf.constant(image2).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Uniform image
    image3 = np.array([[[5, 5, 5], [5, 5, 5]], [[5, 5, 5], [5, 5, 5]]], dtype=np.int32)
    input_dict3 = {"image": tf.constant(image3).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Image with float values
    image4 = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    input_dict4 = {"image": tf.constant(image4).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger image
    image5 = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.int32)
    input_dict5 = {"image": tf.constant(image5).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 4D tensor (batch of images)
    image6 = np.random.randint(0, 256, size=(4, 32, 32, 3), dtype=np.int32)
    input_dict6 = {"image": tf.constant(image6).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 5D tensor
    image7 = np.random.randint(0, 256, size=(2, 4, 32, 32, 3), dtype=np.int32)
    input_dict7 = {"image": tf.constant(image7).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Image with different data type
    image8 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int64)
    input_dict8 = {"image": tf.constant(image8).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Image with small values
    image9 = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.10, 0.11, 0.12]]], dtype=np.float32)
    input_dict9 = {"image": tf.constant(image9).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Image with only one channel
    image10 = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    input_dict10 = {"image": tf.constant(image10).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.per_image_standardization"] = tf_image_per_image_standardization_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.per_image_standardization' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.per_image_standardization'.")

check_valid('tf.image.per_image_standardization', generated_inputs['tf.image.per_image_standardization'], lib="tf", suffix=0)

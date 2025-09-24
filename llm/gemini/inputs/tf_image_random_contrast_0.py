
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_contrast_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D image
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = 0.2
    upper = 0.5
    seed = 123

    input_dict = {
        "image": tf.convert_to_tensor(image).numpy(),
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another 3D image, different contrast range
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32)
    lower = 0.5
    upper = 1.5
    seed = 456

    input_dict = {
        "image": tf.convert_to_tensor(image).numpy(),
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D image (batch of images)
    image = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
                      [[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]]], dtype=np.float32)
    lower = 0.3
    upper = 0.7
    seed = 789

    input_dict = {
        "image": tf.convert_to_tensor(image).numpy(),
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Grayscale image (3D, but last dim is 1)
    image = np.array([[[0.5], [0.6]], [[0.7], [0.8]]], dtype=np.float32)
    lower = 0.8
    upper = 1.2
    seed = 101

    input_dict = {
        "image": tf.convert_to_tensor(image).numpy(),
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5:  3D image with values between 0 and 1
    image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 0.11, 0.12]]], dtype=np.float32)
    lower = 0.9
    upper = 1.1
    seed = 222

    input_dict = {
        "image": tf.convert_to_tensor(image).numpy(),
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 5D image
    image = np.random.rand(2, 2, 2, 2, 3).astype(np.float32)
    lower = 0.4
    upper = 0.6
    seed = 333

    input_dict = {
        "image": tf.convert_to_tensor(image).numpy(),
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Small contrast range
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = 0.99
    upper = 1.01
    seed = 444

    input_dict = {
        "image": tf.convert_to_tensor(image).numpy(),
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Different image size
    image = np.random.rand(32, 32, 3).astype(np.float32)
    lower = 0.7
    upper = 1.3
    seed = 555

    input_dict = {
        "image": tf.convert_to_tensor(image).numpy(),
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another seed
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = 0.2
    upper = 0.5
    seed = 666

    input_dict = {
        "image": tf.convert_to_tensor(image).numpy(),
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Image with different values
    image = np.array([[[100.0, 200.0, 150.0], [50.0, 250.0, 100.0]], [[180.0, 90.0, 220.0], [30.0, 120.0, 200.0]]], dtype=np.float32)
    lower = 0.6
    upper = 0.8
    seed = 777
    input_dict = {
        "image": tf.convert_to_tensor(image).numpy(),
        "lower": lower,
        "upper": upper,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.random_contrast"] = tf_image_random_contrast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.random_contrast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_contrast'.")

check_valid('tf.image.random_contrast', generated_inputs['tf.image.random_contrast'], lib="tf", suffix=0)

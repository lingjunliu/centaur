
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_image_gradients_inputs():
    list_of_inputs = []

    # Input 1: Basic test case
    image = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)
    input_dict = {"image": tf.convert_to_tensor(image).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple channels
    image = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    input_dict = {"image": tf.convert_to_tensor(image).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch size > 1
    image = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
    image = np.concatenate([image, image], axis=0)
    input_dict = {"image": tf.convert_to_tensor(image).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger image dimensions
    image = np.random.rand(1, 10, 10, 1).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different dtype
    image = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float64)
    input_dict = {"image": tf.convert_to_tensor(image).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Image with negative values
    image = np.array([[[[-1], [2]], [[-3], [4]]]], dtype=np.float32)
    input_dict = {"image": tf.convert_to_tensor(image).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch size > 1 with multiple channels
    image = np.random.rand(2, 5, 5, 3).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All zeros
    image = np.zeros((1, 5, 5, 1), dtype=np.float32)
    input_dict = {"image": tf.convert_to_tensor(image).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different image size
    image = np.random.rand(1, 7, 3, 1).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batch of images
    image1 = np.random.rand(1, 4, 4, 1).astype(np.float32)
    image2 = np.random.rand(1, 4, 4, 1).astype(np.float32)
    image = np.concatenate([image1, image2], axis=0)
    input_dict = {"image": tf.convert_to_tensor(image).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.image_gradients"] = tf_image_image_gradients_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.image_gradients' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.image_gradients'.")

check_valid('tf.image.image_gradients', generated_inputs['tf.image.image_gradients'], lib="tf", suffix=0)

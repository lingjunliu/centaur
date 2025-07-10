
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_transpose_inputs():
    list_of_inputs = []

    # Input 1: 3D tensor
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)
    name = "transpose_1"
    input_dict = {"image": tf.convert_to_tensor(image, dtype=tf.int32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D tensor
    image = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    name = "transpose_2"
    input_dict = {"image": tf.convert_to_tensor(image, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor with different shape
    image = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.float64)
    name = "transpose_3"
    input_dict = {"image": tf.convert_to_tensor(image, dtype=tf.float64), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D tensor with different shape
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]], dtype=np.int64)
    name = "transpose_4"
    input_dict = {"image": tf.convert_to_tensor(image, dtype=tf.int64), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor with negative values
    image = np.array([[[1, -2, 3], [-4, 5, -6]], [[-7, 8, -9], [10, -11, 12]]], dtype=np.int32)
    name = "transpose_5"
    input_dict = {"image": tf.convert_to_tensor(image, dtype=tf.int32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D tensor with all same values
    image = np.array([[[[5, 5, 5], [5, 5, 5]], [[5, 5, 5], [5, 5, 5]]]], dtype=np.float32)
    name = "transpose_6"
    input_dict = {"image": tf.convert_to_tensor(image, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor with single channel
    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    name = "transpose_7"
    input_dict = {"image": tf.convert_to_tensor(image, dtype=tf.int32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensor with a batch size of 2
    image = np.array([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]], dtype=np.float32)
    name = "transpose_8"
    input_dict = {"image": tf.convert_to_tensor(image, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with large values
    image = np.array([[[1000, 2000], [3000, 4000]], [[5000, 6000], [7000, 8000]]], dtype=np.int32)
    name = "transpose_9"
    input_dict = {"image": tf.convert_to_tensor(image, dtype=tf.int32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with different data type
    image = np.array([[[[1.5, 2.5], [3.5, 4.5]]]], dtype=np.float64)
    name = "transpose_10"
    input_dict = {"image": tf.convert_to_tensor(image, dtype=tf.float64), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_image_transpose_inputs()
for i in range(len(inputs)):
    inputs[i]['image'] = inputs[i]['image'].numpy()

generated_inputs["tf.image.transpose"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.transpose'.")

check_valid('tf.image.transpose', generated_inputs['tf.image.transpose'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_total_variation_inputs():
    list_of_inputs = []

    # Input 1: 3D image
    images = np.random.rand(32, 32, 3).astype(np.float32)
    name = "image1"
    input_dict = {"images": tf.convert_to_tensor(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D image batch
    images = np.random.rand(2, 64, 64, 1).astype(np.float32)
    name = "image2"
    input_dict = {"images": tf.convert_to_tensor(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D image with a single channel
    images = np.random.rand(16, 16, 1).astype(np.float32)
    name = "image3"
    input_dict = {"images": tf.convert_to_tensor(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D image batch with larger batch size
    images = np.random.rand(10, 28, 28, 3).astype(np.float32)
    name = "image4"
    input_dict = {"images": tf.convert_to_tensor(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D image with different dimensions
    images = np.random.rand(48, 32, 4).astype(np.float32)
    name = "image5"
    input_dict = {"images": tf.convert_to_tensor(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D image with different channel
    images = np.random.rand(1, 100, 100, 4).astype(np.float32)
    name = "image6"
    input_dict = {"images": tf.convert_to_tensor(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D image with small dimensions
    images = np.random.rand(2, 2, 3).astype(np.float32)
    name = "image7"
    input_dict = {"images": tf.convert_to_tensor(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D image batch with small dimensions
    images = np.random.rand(3, 5, 5, 1).astype(np.float32)
    name = "image8"
    input_dict = {"images": tf.convert_to_tensor(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D image with a specified name
    images = np.random.rand(20, 20, 3).astype(np.float32)
    name = "specified_name"
    input_dict = {"images": tf.convert_to_tensor(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D image batch with a specified name
    images = np.random.rand(4, 30, 30, 1).astype(np.float32)
    name = "another_name"
    input_dict = {"images": tf.convert_to_tensor(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = np.random.randn(1, 25, 25, 3).astype(np.float32)
    name = "negative_image"
    input_dict = {"images": tf.convert_to_tensor(images), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
temp_list = tf_image_total_variation_inputs()
for i in range(len(temp_list)):
  temp_list[i]['images'] = temp_list[i]['images'].numpy()

generated_inputs["tf.image.total_variation"] = temp_list

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.total_variation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.total_variation'.")

check_valid('tf.image.total_variation', generated_inputs['tf.image.total_variation'], lib="tf", suffix=0)

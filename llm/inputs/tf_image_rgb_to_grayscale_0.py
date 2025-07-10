
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_rgb_to_grayscale_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D image
    images = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    name = "basic_image"
    input_dict = {"images": tf.convert_to_tensor(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of 2D images
    images = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)
    name = "batch_image"
    input_dict = {"images": tf.convert_to_tensor(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D image
    images = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    name = "3d_image"
    input_dict = {"images": tf.convert_to_tensor(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D image (batch of 3D images)
    images = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
                       [[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]]], dtype=np.float32)
    name = "4d_image"
    input_dict = {"images": tf.convert_to_tensor(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  Image with different values
    images = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    name = "diff_values"
    input_dict = {"images": tf.convert_to_tensor(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  Empty name
    images = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    name = ""
    input_dict = {"images": tf.convert_to_tensor(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Image with different dimensions
    images = np.array([[[[1.0, 2.0, 3.0]]]], dtype=np.float32)
    name = "different_dimensions"
    input_dict = {"images": tf.convert_to_tensor(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Image with string name
    images = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    name = "image_name_1"
    input_dict = {"images": tf.convert_to_tensor(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Image with integer type, convert to float32
    images = np.array([[[1, 2, 3]]], dtype=np.int32).astype(np.float32)
    name = "int_type"
    input_dict = {"images": tf.convert_to_tensor(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Image with float64 type, convert to float32
    images = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64).astype(np.float32)
    name = "float64_type"
    input_dict = {"images": tf.convert_to_tensor(images).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.rgb_to_grayscale"] = tf_image_rgb_to_grayscale_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.rgb_to_grayscale' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.rgb_to_grayscale'.")

check_valid('tf.image.rgb_to_grayscale', generated_inputs['tf.image.rgb_to_grayscale'], lib="tf", suffix=0)

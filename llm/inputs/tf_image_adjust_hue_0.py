
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_adjust_hue_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    delta = 0.2
    name = "adjust_hue_1"
    input_dict = {"image": image, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative delta
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    delta = -0.3
    name = "adjust_hue_2"
    input_dict = {"image": image, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Delta at the boundary
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    delta = 1.0
    name = "adjust_hue_3"
    input_dict = {"image": image, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Delta at the boundary
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    delta = -1.0
    name = "adjust_hue_4"
    input_dict = {"image": image, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D image
    image = np.random.rand(2, 3, 3).astype(np.float32)
    delta = 0.5
    name = "adjust_hue_5"
    input_dict = {"image": image, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single pixel image
    image = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
    delta = 0.7
    name = "adjust_hue_6"
    input_dict = {"image": image, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Larger image
    image = np.random.rand(100, 100, 3).astype(np.float32)
    delta = 0.1
    name = "adjust_hue_7"
    input_dict = {"image": image, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Image with 4 dimensions
    image = np.random.rand(1, 20, 20, 3).astype(np.float32)
    delta = -0.8
    name = "adjust_hue_8"
    input_dict = {"image": image, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different float type
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float64)
    delta = 0.4
    name = "adjust_hue_9"
    input_dict = {"image": image, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero delta
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    delta = 0.0
    name = "adjust_hue_10"
    input_dict = {"image": image, "delta": delta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.adjust_hue"] = tf_image_adjust_hue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.adjust_hue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_hue'.")

check_valid('tf.image.adjust_hue', generated_inputs['tf.image.adjust_hue'], lib="tf", suffix=0)

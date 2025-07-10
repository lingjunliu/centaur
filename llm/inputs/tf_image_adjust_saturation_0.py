
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_adjust_saturation_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D RGB image
    image = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    saturation_factor = 0.5
    name = "adjust1"
    input_dict = {"image": image, "saturation_factor": saturation_factor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 2D RGB image, saturation factor > 1
    image = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    saturation_factor = 2.0
    name = "adjust2"
    input_dict = {"image": image, "saturation_factor": saturation_factor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D RGB images
    image = np.random.rand(2, 3, 4, 3).astype(np.float32)
    saturation_factor = 0.75
    name = "adjust3"
    input_dict = {"image": image, "saturation_factor": saturation_factor, "name": name}
    image = image.reshape((2 * 3, 4, 3))
    input_dict["image"] = image
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single pixel image
    image = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
    saturation_factor = 1.5
    name = "adjust4"
    input_dict = {"image": image, "saturation_factor": saturation_factor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  Image with values outside [0,1]
    image = np.array([[[1.5, -0.5, 0.2]]], dtype=np.float32)
    saturation_factor = 0.8
    name = "adjust5"
    input_dict = {"image": image, "saturation_factor": saturation_factor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Image with zero saturation
    image = np.array([[[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]]], dtype=np.float32)
    saturation_factor = 0.0
    name = "adjust6"
    input_dict = {"image": image, "saturation_factor": saturation_factor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger image, different dtype
    image = np.random.rand(100, 100, 3).astype(np.float32)
    saturation_factor = 1.2
    name = "adjust7"
    input_dict = {"image": image, "saturation_factor": saturation_factor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D RGB images, smaller saturation factor
    image = np.random.rand(1, 2, 2, 3).astype(np.float32)
    saturation_factor = 0.25
    name = "adjust8"
    image = image.reshape((2, 2, 3))
    input_dict = {"image": image, "saturation_factor": saturation_factor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Image with identical channels (grayscale approximation)
    image = np.array([[[0.5, 0.5, 0.5], [0.7, 0.7, 0.7]], [[0.2, 0.2, 0.2], [0.9, 0.9, 0.9]]], dtype=np.float32)
    saturation_factor = 1.0
    name = "adjust9"
    input_dict = {"image": image, "saturation_factor": saturation_factor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Simple B&W image
    image = np.array([[[0., 0., 0.], [1., 1., 1.]]], dtype=np.float32)
    saturation_factor = 1.5
    name = "adjust10"
    input_dict = {"image": image, "saturation_factor": saturation_factor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.image.adjust_saturation"] = tf_image_adjust_saturation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.image.adjust_saturation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_saturation'.")

check_valid('tf.image.adjust_saturation', generated_inputs['tf.image.adjust_saturation'], lib="tf", suffix=0)

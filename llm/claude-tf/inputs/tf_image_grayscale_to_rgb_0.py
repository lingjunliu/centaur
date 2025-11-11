
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_image_grayscale_to_rgb_inputs():
    list_of_inputs = []
    
    images = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    name = None
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.array([[[[0.5]], [[0.7]], [[0.9]]]], dtype=np.float32)
    name = "grayscale_conversion"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.random.rand(1, 4, 4, 1).astype(np.float32)
    name = None
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.random.rand(3, 2, 2, 1).astype(np.float32)
    name = "batch_conversion"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.array([[[[100], [150]], [[200], [250]]]], dtype=np.uint8)
    name = None
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.array([[[[0.1], [0.2], [0.3]]]], dtype=np.float64)
    name = "float64_conversion"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.array([[[[128]]]], dtype=np.uint8)
    name = None
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.random.rand(5, 8, 8, 1).astype(np.float32)
    name = "large_batch"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.zeros((1, 3, 3, 1), dtype=np.float32)
    name = None
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.ones((2, 5, 5, 1), dtype=np.float32)
    name = "ones_image"
    input_dict = {"images": images, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.grayscale_to_rgb"] = tf_image_grayscale_to_rgb_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.grayscale_to_rgb' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.grayscale_to_rgb'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.grayscale_to_rgb', generated_inputs['tf.image.grayscale_to_rgb'], lib="tf", suffix=0)

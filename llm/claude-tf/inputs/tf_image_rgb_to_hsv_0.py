
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_image_rgb_to_hsv_inputs():
    list_of_inputs = []
    
    images = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
    input_dict = {"images": images, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    input_dict = {"images": images, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.stack([np.zeros((5, 5)), np.zeros((5, 5)), np.ones((5, 5))], axis=-1).astype(np.float32)
    input_dict = {"images": images, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.array([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.3, 0.4, 0.5]]]], dtype=np.float32)
    input_dict = {"images": images, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.array([[[0.3, 0.7, 0.2]]], dtype=np.float64)
    input_dict = {"images": images, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.zeros((3, 3, 3), dtype=np.float32)
    input_dict = {"images": images, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.ones((2, 2, 3), dtype=np.float32)
    input_dict = {"images": images, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float16)
    input_dict = {"images": images, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.array([[[[0.2, 0.3, 0.4]]]], dtype=np.float32)
    input_dict = {"images": images, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.array([[[0.8, 0.2, 0.5], [0.1, 0.9, 0.3]], [[0.6, 0.4, 0.7], [0.2, 0.3, 0.8]]], dtype=np.float32)
    input_dict = {"images": images, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.rgb_to_hsv"] = tf_image_rgb_to_hsv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.rgb_to_hsv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.rgb_to_hsv'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.rgb_to_hsv', generated_inputs['tf.image.rgb_to_hsv'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_image_sobel_edges_inputs():
    list_of_inputs = []
    
    image = np.random.uniform(0, 255, size=(1, 2, 2, 3)).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.uniform(0, 255, size=(1, 28, 28, 3)).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.uniform(0, 255, size=(1, 32, 32, 1)).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.uniform(0, 255, size=(4, 64, 64, 3)).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.uniform(0, 255, size=(1, 50, 50, 3)).astype(np.float64)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.uniform(0, 255, size=(10, 10, 10, 1)).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.uniform(0, 255, size=(2, 16, 16, 4)).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.uniform(0, 255, size=(1, 20, 100, 3)).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.uniform(0, 255, size=(1, 100, 20, 3)).astype(np.float32)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.uniform(0, 255, size=(2, 128, 128, 3)).astype(np.float64)
    input_dict = {"image": image}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.sobel_edges"] = tf_image_sobel_edges_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.sobel_edges' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.sobel_edges'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.sobel_edges', generated_inputs['tf.image.sobel_edges'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def sobel_edges_inputs():
    list_of_inputs = []
    
    # Input 1: Basic 3D tensor (batch_size=1, height=28, width=28, channels=3)
    image = np.random.uniform(0, 255, [1, 28, 28, 3]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with batch_size=2
    image = np.random.uniform(0, 255, [2, 16, 16, 3]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 4D tensor with batch_size=1, height=32, width=32, channel=1
    image = np.random.uniform(0, 255, [1, 32, 32, 1]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with batch_size=1, height=64, width=64, channel=4
    image = np.random.uniform(0, 255, [1, 64, 64, 4]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D tensor with batch_size=1, height=10, width=10, channel=3
    image = np.random.uniform(0, 255, [1, 10, 10, 3]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with batch_size=1, height=32, width=32, channel=5
    image = np.random.uniform(0, 255, [1, 32, 32, 5]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor with batch_size=3, height=8, width=8, channel=3
    image = np.random.uniform(0, 255, [3, 8, 8, 3]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with batch_size=1, height=20, width=20, channel=6
    image = np.random.uniform(0, 255, [1, 20, 20, 6]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor with batch_size=1, height=15, width=15, channel=2
    image = np.random.uniform(0, 255, [1, 15, 15, 2]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with batch_size=1, height=30, width=30, channel=7
    image = np.random.uniform(0, 255, [1, 30, 30, 7]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.sobel_edges"] = sobel_edges_inputs()

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

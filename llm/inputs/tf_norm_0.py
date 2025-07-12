
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_norm_inputs():
    list_of_inputs = []

    # Input 1: Vector norm, default ord, no axis
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {'tensor': tensor, 'ord': 'euclidean', 'axis': None, 'keepdims': False, 'name': 'norm1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix norm, Frobenius, no axis (treated as flattened vector)
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {'tensor': tensor, 'ord': 'fro', 'axis': None, 'keepdims': False, 'name': 'norm2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Vector norm, 1-norm, axis=0
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {'tensor': tensor, 'ord': '1', 'axis': (0,), 'keepdims': False, 'name': 'norm3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.norm"] = tf_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.norm'.")

check_valid('tf.norm', generated_inputs['tf.norm'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_normalize_inputs():
    list_of_inputs = []

    # Input 1: Vector normalization with euclidean norm
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ord = 'euclidean'
    axis = None
    name = 'normalize_vector'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Matrix normalization with frobenius norm
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'fro'
    axis = None
    name = 'normalize_matrix'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Vector normalization along axis 0
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = (0,)
    name = 'normalize_axis_0'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Vector normalization along axis 1
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = (1,)
    name = 'normalize_axis_1'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Matrix normalization using 1-norm
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = '1'
    axis = (0, 1)
    name = 'normalize_1_norm'
    input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.normalize"] = tf_linalg_normalize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.normalize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.normalize'.")

check_valid('tf.linalg.normalize', generated_inputs['tf.linalg.normalize'], lib="tf", suffix=0)

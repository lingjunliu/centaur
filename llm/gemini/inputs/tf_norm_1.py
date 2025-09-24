
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_norm_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ord = 'euclidean'
    axis = None
    keepdims = False
    name = 'norm_1'

    input_dict = {
        "tensor": tensor,
        "ord": ord,
        "axis": axis,
        "keepdims": keepdims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = None
    keepdims = False
    name = 'norm_2'

    input_dict = {
        "tensor": tensor,
        "ord": ord,
        "axis": axis,
        "keepdims": keepdims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = 0
    keepdims = False
    name = 'norm_3'

    input_dict = {
        "tensor": tensor,
        "ord": ord,
        "axis": axis,
        "keepdims": keepdims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = 1
    keepdims = False
    name = 'norm_4'

    input_dict = {
        "tensor": tensor,
        "ord": ord,
        "axis": axis,
        "keepdims": keepdims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = 0
    keepdims = True
    name = 'norm_5'

    input_dict = {
        "tensor": tensor,
        "ord": ord,
        "axis": axis,
        "keepdims": keepdims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'euclidean'
    axis = 1
    keepdims = True
    name = 'norm_6'

    input_dict = {
        "tensor": tensor,
        "ord": ord,
        "axis": axis,
        "keepdims": keepdims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = 'euclidean'
    axis = 0
    keepdims = False
    name = 'norm_7'

    input_dict = {
        "tensor": tensor,
        "ord": ord,
        "axis": axis,
        "keepdims": keepdims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = 'euclidean'
    axis = None
    keepdims = False
    name = 'norm_8'

    input_dict = {
        "tensor": tensor,
        "ord": ord,
        "axis": axis,
        "keepdims": keepdims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    ord = 'euclidean'
    axis = None
    keepdims = False
    name = 'norm_9'

    input_dict = {
        "tensor": tensor,
        "ord": ord,
        "axis": axis,
        "keepdims": keepdims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = 'euclidean'
    axis = 1
    keepdims = False
    name = 'norm_10'

    input_dict = {
        "tensor": tensor,
        "ord": ord,
        "axis": axis,
        "keepdims": keepdims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.norm_1"] = tf_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.norm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.norm_1'.")

check_valid('tf.norm', generated_inputs['tf.norm_1'], lib="tf", suffix=1)

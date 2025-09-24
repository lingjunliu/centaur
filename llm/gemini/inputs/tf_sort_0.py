
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sort_inputs():
    list_of_inputs = []

    # Input 1
    values = np.array([1, 10, 26, 2, 166, 62], dtype=np.int32)
    axis = -1
    direction = 'ASCENDING'
    name = None
    input_dict = {'values': values, 'axis': axis, 'direction': direction, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    values = np.array([1.0, 10.0, 26.0, 2.0, 166.0, 62.0], dtype=np.float32)
    axis = -1
    direction = 'DESCENDING'
    name = "sort_desc"
    input_dict = {'values': values, 'axis': axis, 'direction': direction, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multidimensional, axis=0
    values = np.array([[3, 2, 1], [2, 1, 3], [1, 3, 2]], dtype=np.int32)
    axis = 0
    direction = 'ASCENDING'
    name = None
    input_dict = {'values': values, 'axis': axis, 'direction': direction, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multidimensional, axis=1 (equivalent to axis=-1)
    values = np.array([[3, 2, 1], [2, 1, 3], [1, 3, 2]], dtype=np.int32)
    axis = 1
    direction = 'ASCENDING'
    name = None
    input_dict = {'values': values, 'axis': axis, 'direction': direction, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor
    values = np.array([[[3, 2, 1], [2, 1, 3]], [[1, 3, 2], [4, 5, 6]]], dtype=np.int32)
    axis = -1
    direction = 'ASCENDING'
    name = None
    input_dict = {'values': values, 'axis': axis, 'direction': direction, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor, descending
    values = np.array([[[3, 2, 1], [2, 1, 3]], [[1, 3, 2], [4, 5, 6]]], dtype=np.int32)
    axis = 0
    direction = 'DESCENDING'
    name = None
    input_dict = {'values': values, 'axis': axis, 'direction': direction, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor, different axis
    values = np.array([[[3, 2, 1], [2, 1, 3]], [[1, 3, 2], [4, 5, 6]]], dtype=np.int32)
    axis = 1
    direction = 'ASCENDING'
    name = None
    input_dict = {'values': values, 'axis': axis, 'direction': direction, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D tensor with negative values
    values = np.array([-1, -10, -26, -2, -166, -62], dtype=np.int32)
    axis = -1
    direction = 'ASCENDING'
    name = None
    input_dict = {'values': values, 'axis': axis, 'direction': direction, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D tensor with negative values
    values = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    axis = -1
    direction = 'ASCENDING'
    name = None
    input_dict = {'values': values, 'axis': axis, 'direction': direction, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64
    values = np.array([1.0, 10.0, 26.0, 2.0, 166.0, 62.0], dtype=np.float64)
    axis = -1
    direction = 'ASCENDING'
    name = None
    input_dict = {'values': values, 'axis': axis, 'direction': direction, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sort"] = tf_sort_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.sort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sort'.")

check_valid('tf.sort', generated_inputs['tf.sort'], lib="tf", suffix=0)

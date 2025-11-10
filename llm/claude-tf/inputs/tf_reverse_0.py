
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_reverse_inputs():
    list_of_inputs = []
    
    tensor = np.array([1, 2, 3, 4, 5])
    axis = [0]
    name = "reverse_1d"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[1, 2, 3], [4, 5, 6]])
    axis = [0]
    name = "reverse_2d_axis0"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[1, 2, 3], [4, 5, 6]])
    axis = [1]
    name = "reverse_2d_axis1"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = [-1]
    name = "reverse_3d_negative_axis"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]])
    axis = [3]
    name = "reverse_4d"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    axis = [0, 1]
    name = "reverse_multiple_axes"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
    axis = [0]
    name = "reverse_float"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([True, False, True, False], dtype=np.bool_)
    axis = [0]
    name = "reverse_bool"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[1, 2], [3, 4]])
    axis = []
    name = "reverse_empty_axis"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int32)
    axis = [-2]
    name = "reverse_negative_axis2"
    input_dict = {"tensor": tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.reverse"] = tf_reverse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.reverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.reverse'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.reverse', generated_inputs['tf.reverse'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_expand_dims_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor, axis=0
    input_tensor = np.array([1, 2, 3])
    axis = 0
    name = "expand_dim_1"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, axis=1
    input_tensor = np.array([[1, 2], [3, 4]])
    axis = 1
    name = "expand_dim_2"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, axis=-1
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = -1
    name = "expand_dim_3"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor, axis=-1
    input_tensor = np.array([1, 2, 3, 4])
    axis = -1
    name = "expand_dim_4"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar tensor, axis=0
    input_tensor = np.array(5)
    axis = 0
    name = "expand_dim_5"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensor, axis=0
    input_tensor = np.array([[1, 2, 3]])
    axis = 0
    name = "expand_dim_6"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor, axis=1
    input_tensor = np.random.rand(2, 3, 4)
    axis = 1
    name = "expand_dim_7"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor, axis=-2
    input_tensor = np.random.rand(2, 3, 4, 5)
    axis = -2
    name = "expand_dim_8"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor, axis=0
    input_tensor = np.random.rand(2, 3, 4, 5)
    axis = 0
    name = "expand_dim_9"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D tensor, axis=-3
    input_tensor = np.array([[1, 2], [3, 4]])
    axis = -3
    name = "expand_dim_10"
    input_dict = {"input": input_tensor, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.expand_dims"] = tf_expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.expand_dims' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.expand_dims'.")

check_valid('tf.expand_dims', generated_inputs['tf.expand_dims'], lib="tf", suffix=0)

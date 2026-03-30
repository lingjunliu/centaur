
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_accumulate_n_inputs():
    list_of_inputs = []

    # Input 1
    inputs = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 0], [0, 6]], dtype=np.int32)]
    shape = [2, 2]
    tensor_dtype = np.int32
    name = "accumulate_1"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32), np.array([7.0, 8.0, 9.0], dtype=np.float32)]
    shape = [3]
    tensor_dtype = np.float32
    name = "accumulate_2"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = [np.array([[-1, -2], [-3, -4]], dtype=np.int64), np.array([[5, 0], [0, 6]], dtype=np.int64)]
    shape = [2, 2]
    tensor_dtype = np.int64
    name = "accumulate_3"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = [np.array([1, 2, 3], dtype=np.int16), np.array([4, 5, 6], dtype=np.int16), np.array([7, 8, 9], dtype=np.int16)]
    shape = [3]
    tensor_dtype = np.int16
    name = "accumulate_4"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = [np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64), np.array([[0.5, 0.0], [0.0, 0.6]], dtype=np.float64)]
    shape = [2, 2]
    tensor_dtype = np.float64
    name = "accumulate_5"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}

def convert_input_to_numpy(input_data):
  if isinstance(input_data, list):
    numpy_list = []
    for item in input_data:
      numpy_list.append(np.array(item))
    return numpy_list
  else:
    return np.array(input_data)

temp_inputs = tf_math_accumulate_n_inputs()

for input_dict in temp_inputs:
    input_dict["inputs"] = [np.array(x) for x in input_dict["inputs"]]

generated_inputs["tf.math.accumulate_n"] = temp_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.accumulate_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.accumulate_n'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.accumulate_n', generated_inputs['tf.math.accumulate_n'], lib="tf", suffix=0)

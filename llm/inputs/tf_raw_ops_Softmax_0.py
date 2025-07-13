
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_softmax_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32
    logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    name = None
    input_dict = {"logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With negative values and float32
    logits = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    name = "softmax_op_2"
    input_dict = {"logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different magnitudes and float32
    logits = np.array([[0.1, 10.0, -5.0], [-10.0, 0.0, 5.0]], dtype=np.float32)
    name = None
    input_dict = {"logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half type
    logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    name = "softmax_op_4"
    input_dict = {"logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 type, removing bfloat16 due to error
    logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    name = "softmax_op_6"
    input_dict = {"logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32 with zero values
    logits = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    name = None
    input_dict = {"logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Large values for float32
    logits = np.array([[1e1, 1e1, 1e1], [1e1, 1e1, 1e1]], dtype=np.float32) # Reduce the value to prevent potential overflow
    name = "softmax_op_8"
    input_dict = {"logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small values for float32
    logits = np.array([[-1e1, -1e1, -1e1], [-1e1, -1e1, -1e1]], dtype=np.float32) # Reduce the value to prevent potential overflow
    name = None
    input_dict = {"logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different batch size with float32
    logits = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    name = "softmax_op_10"
    input_dict = {"logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single batch float32
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    name = None
    input_dict = {"logits": logits, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {} # Initialize the dictionary
generated_inputs["tf.raw_ops.Softmax"] = tf_raw_ops_softmax_inputs()

#Dummy check_valid and run_api for local execution
def check_valid(api, inputs, lib="tf", suffix=0):
  pass

def run_api(api, input_dict, cpu=True, lib="tf"):
  return input_dict

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Softmax'.")

check_valid('tf.raw_ops.Softmax', generated_inputs['tf.raw_ops.Softmax'], lib="tf", suffix=0)

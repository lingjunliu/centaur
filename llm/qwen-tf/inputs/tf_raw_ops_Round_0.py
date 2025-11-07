
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_round_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with positive values
    x = np.array([1.4, 2.7, 3.1, 4.9], dtype=np.float32)
    input_dict = {"name": "round_test_1", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 2: float64 tensor with negative values
    x = np.array([-1.5, -2.7, -3.1, -4.9], dtype=np.float64)
    input_dict = {"name": "round_test_2", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 3: int32 tensor with mixed values
    x = np.array([1, -2, 3, -4], dtype=np.int32)
    input_dict = {"name": "round_test_3", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 4: int64 tensor with positive values
    x = np.array([10, 20, 30, 40], dtype=np.int64)
    input_dict = {"name": "round_test_4", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 5: float32 tensor with decimal values
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    input_dict = {"name": "round_test_5", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 6: bfloat16 tensor with positive values
    x = np.array([1.1, 2.2, 3.3], dtype=np.float32)  # Using float32 for bfloat16
    input_dict = {"name": "round_test_6", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 7: half tensor with negative values
    x = np.array([-1.1, -2.2, -3.3], dtype=np.float16)
    input_dict = {"name": "round_test_7", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 8: int16 tensor with mixed values
    x = np.array([1, -2, 3, -4], dtype=np.int16)
    input_dict = {"name": "round_test_8", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 9: int8 tensor with positive values
    x = np.array([1, 2, 3, 4], dtype=np.int8)
    input_dict = {"name": "round_test_9", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 10: float32 tensor with decimal values
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    input_dict = {"name": "round_test_10", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict)

    return list_of_inputs

generated_inputs["tf.raw_ops.Round"] = generate_round_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Round'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Round', generated_inputs['tf.raw_ops.Round'], lib="tf", suffix=0)

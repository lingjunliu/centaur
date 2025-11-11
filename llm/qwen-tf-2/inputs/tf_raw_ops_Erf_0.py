
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_erf_inputs():
    list_of_inputs = []
    
    # Input 1: Single element tensor
    x = np.array([1.0], dtype=np.float32)
    input_dict = {'name': 'test_1', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor with positive values
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {'name': 'test_2', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with negative values
    x = np.array([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]], dtype=np.float32)
    input_dict = {'name': 'test_3', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Mixed positive and negative values
    x = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]], dtype=np.float32)
    input_dict = {'name': 'test_4', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single negative value
    x = np.array([-1.0], dtype=np.float32)
    input_dict = {'name': 'test_5', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large values
    x = np.array([10.0, 20.0], dtype=np.float32)
    input_dict = {'name': 'test_6', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Float64 type
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {'name': 'test_7', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Half type
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_dict = {'name': 'test_8', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Bfloat16 type
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = x.astype(np.float16)  # This simulates bfloat16
    input_dict = {'name': 'test_9', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Very small values
    x = np.array([0.001, 0.0001], dtype=np.float32)
    input_dict = {'name': 'test_10', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Erf"] = generate_erf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Erf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Erf', generated_inputs['tf.raw_ops.Erf'], lib="tf", suffix=0)

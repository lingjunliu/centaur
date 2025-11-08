
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_log1p_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with positive values
    x = np.array([0., 0.5, 1., 5.], dtype=np.float32)
    input_dict = {"x": x, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with positive values
    x = np.array([0., 0.5, 1., 5.], dtype=np.float64)
    input_dict = {"x": x, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: float32 tensor with negative values
    x = np.array([-0.9, -0.5, 0., 0.5], dtype=np.float32)
    input_dict = {"x": x, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: float64 tensor with negative values
    x = np.array([-0.9, -0.5, 0., 0.5], dtype=np.float64)
    input_dict = {"x": x, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: complex64 tensor with positive values
    x = np.array([1.+0j, 2.+0j, 3.+0j], dtype=np.complex64)
    input_dict = {"x": x, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: complex128 tensor with positive values
    x = np.array([1.+0j, 2.+0j, 3.+0j], dtype=np.complex128)
    input_dict = {"x": x, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: half tensor with positive values
    x = np.array([0., 0.5, 1., 5.], dtype=np.float16)
    input_dict = {"x": x, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: bfloat16 tensor with positive values
    x = np.array([0., 0.5, 1., 5.], dtype=np.float32)
    input_dict = {"x": x, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: scalar tensor (1D array) with positive values
    x = np.array([0.5], dtype=np.float32)
    input_dict = {"x": x, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with positive values
    x = np.array([[0., 0.5], [1., 5.]], dtype=np.float32)
    input_dict = {"x": x, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.log1p"] = tf_math_log1p_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.log1p'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.log1p', generated_inputs['tf.math.log1p'], lib="tf", suffix=0)

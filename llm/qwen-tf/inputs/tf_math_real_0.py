
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_real_inputs():
    list_of_inputs = []
    
    # Input 1, valid - real tensor
    input_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict_1 = {"input": input_1, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    
    # Input 2, valid - complex tensor with real part
    input_2 = np.array([1+2j, 3+4j], dtype=np.complex64)
    input_dict_2 = {"input": input_2, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    
    # Input 3, valid - complex tensor with negative real part
    input_3 = np.array([-1-2j, -3-4j], dtype=np.complex64)
    input_dict_3 = {"input": input_3, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4, valid - complex tensor with mixed real and imaginary parts
    input_4 = np.array([1.5+2.5j, 3.7+4.8j], dtype=np.complex64)
    input_dict_4 = {"input": input_4, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5, valid - real tensor with negative values
    input_5 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict_5 = {"input": input_5, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6, valid - complex tensor with zero imaginary part
    input_6 = np.array([1.0+0j, 2.0+0j], dtype=np.complex64)
    input_dict_6 = {"input": input_6, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7, valid - complex tensor with real part in float64
    input_7 = np.array([1.0+2.0j, 3.0+4.0j], dtype=np.complex128)
    input_dict_7 = {"input": input_7, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8, valid - real tensor with multiple dimensions
    input_8 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict_8 = {"input": input_8, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9, valid - complex tensor with real part in float64
    input_9 = np.array([1.0+2.0j, 3.0+4.0j], dtype=np.complex128)
    input_dict_9 = {"input": input_9, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10, valid - real tensor with complex numbers
    input_10 = np.array([1.0+0j, 2.0+0j], dtype=np.complex64)
    input_dict_10 = {"input": input_10, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs["tf.math.real"] = tf_math_real_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.real'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.real', generated_inputs['tf.math.real'], lib="tf", suffix=0)

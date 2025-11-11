
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_real_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "real_part"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    input_dict = {"input": input_tensor, "name": "real_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "already_real"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-5-3j, -2+1j, -7-9j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "negative_real"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex128)
    input_dict = {"input": input_tensor, "name": "complex_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array(3+4j, dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "scalar_complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-1.0, -2.5, -3.7], dtype=np.float64)
    input_dict = {"input": input_tensor, "name": "negative_floats"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([5+0j, 10+0j, 15+0j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "zero_imaginary"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": "integers"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0+1j, 0+2j, 0+3j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "name": "zero_real"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "large_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
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

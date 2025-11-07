
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_pow_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensors
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    y = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "pow1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: int64 tensors
    x = np.array([[2, 3], [4, 5]], dtype=np.int64)
    y = np.array([[2, 3], [4, 5]], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "pow2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: float64 tensors
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    y = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "pow3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: complex64 tensors
    x = np.array([[2+3j, 4+5j], [6+7j, 8+9j]], dtype=np.complex64)
    y = np.array([[2+3j, 4+5j], [6+7j, 8+9j]], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "pow4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: complex128 tensors
    x = np.array([[2+3j, 4+5j], [6+7j, 8+9j]], dtype=np.complex128)
    y = np.array([[2+3j, 4+5j], [6+7j, 8+9j]], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": "pow5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: negative values
    x = np.array([[-2, -3], [-4, -5]], dtype=np.float32)
    y = np.array([[2, 3], [4, 5]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "pow6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: one-dimensional tensors
    x = np.array([2, 3, 4], dtype=np.float32)
    y = np.array([2, 3, 4], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "pow7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: scalar tensors
    x = np.array(2.0, dtype=np.float32)
    y = np.array(3.0, dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "pow8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: zero tensors
    x = np.array([[0, 2], [3, 4]], dtype=np.float32)
    y = np.array([[0, 3], [4, 5]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "pow9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: mixed types (int32 and float64)
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "pow10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.pow"] = tf_math_pow_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.pow'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.pow', generated_inputs['tf.math.pow'], lib="tf", suffix=0)

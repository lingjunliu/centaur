
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_math_floor_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with positive values
    x = np.array([1.3324, -1.5, 5.555, -2.532, 0.99, float("inf")], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "floor_1"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: float64 tensor with negative values
    x = np.array([-1.3324, -1.5, -5.555, -2.532, -0.99, float("-inf")], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "floor_2"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: bfloat16 tensor with mixed values
    x = np.array([1.33, -1.5, 5.55, -2.53, 0.99, float("inf")], dtype=np.float16)
    input_dict = {
        "x": x,
        "name": "floor_3"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: float32 tensor with one dimension
    x = np.array([1.33, -1.5, 5.55, -2.53, 0.99], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "floor_4"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: float64 tensor with two dimensions
    x = np.array([[1.33, -1.5], [5.55, -2.53]], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "floor_5"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: half tensor with one dimension
    x = np.array([1.33, -1.5, 5.55, -2.53, 0.99], dtype=np.float16)
    input_dict = {
        "x": x,
        "name": "floor_6"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: float32 tensor with scalar value
    x = np.array(1.33, dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "floor_7"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: float64 tensor with negative values
    x = np.array([-1.33, -1.5, -5.55, -2.53], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "floor_8"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: float32 tensor with decimal values
    x = np.array([1.33, -1.5, 5.55, -2.53, 0.99], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "floor_9"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: bfloat16 tensor with large values
    x = np.array([1.33, -1.5, 5.55, -2.53, 0.99], dtype=np.float16)
    input_dict = {
        "x": x,
        "name": "floor_10"
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.math.floor"] = tf_math_floor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.floor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.floor', generated_inputs['tf.math.floor'], lib="tf", suffix=0)

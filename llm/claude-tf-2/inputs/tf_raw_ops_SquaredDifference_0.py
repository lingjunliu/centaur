
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_squared_difference_inputs():
    list_of_inputs = []
    
    input_dict = {
        "name": "squared_diff_1d",
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "y": np.array([0.5, 1.5, 2.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "squared_diff_2d",
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "y": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "squared_diff_negative",
        "x": np.array([-5, -3, 0, 3, 5], dtype=np.int32),
        "y": np.array([2, -1, 0, -2, 1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "squared_diff_3d",
        "x": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        "y": np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "squared_diff_broadcast",
        "x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "y": np.array([2.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "squared_diff_int64",
        "x": np.array([100, 200, 300], dtype=np.int64),
        "y": np.array([50, 150, 250], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "squared_diff_half",
        "x": np.array([0.5, 1.5, 2.5], dtype=np.float16),
        "y": np.array([0.25, 1.25, 2.25], dtype=np.float16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "squared_diff_complex64",
        "x": np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64),
        "y": np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "squared_diff_complex128",
        "x": np.array([1+2j, 3+4j], dtype=np.complex128),
        "y": np.array([0+1j, 1+1j], dtype=np.complex128)
    }
    list_of_inputs.append(copy.deepcopy(

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SquaredDifference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SquaredDifference'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SquaredDifference', generated_inputs['tf.raw_ops.SquaredDifference'], lib="tf", suffix=0)

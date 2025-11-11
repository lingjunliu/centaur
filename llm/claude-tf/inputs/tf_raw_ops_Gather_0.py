
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_gather_inputs():
    list_of_inputs = []
    
    params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    indices = np.array(0, dtype=np.int32)
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": True,
        "name": "gather_op_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    params = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
    indices = np.array([0, 2, 1], dtype=np.int32)
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": False,
        "name": "gather_op_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    params = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float64)
    indices = np.array([2, 0], dtype=np.int64)
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": True,
        "name": "gather_op_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    params = np.array([100, 200, 300, 400, 500], dtype=np.int32)
    indices = np.array([4, 2, 0], dtype=np.int32)
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": True,
        "name": "gather_op_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
    indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": False,
        "name": "gather_op_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    params = np.random.randn(5, 3, 4, 2).astype(np.float32)
    indices = np.array(3, dtype=np.int64)
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": True,
        "name": "gather_op_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    params = np.arange(100, dtype=np.float32)
    indices = np.array([10, 20, 30, 50, 99], dtype=np.int32)
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": True,
        "name": "gather_op_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    params = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=np.float32)
    indices = np.array([1], dtype=np.int32)
    input_dict = {
        "params": params,
        "indices": indices,
        "validate_indices": False,
        "name": "gather_op_8"
    }
    list_

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Gather' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Gather'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Gather', generated_inputs['tf.raw_ops.Gather'], lib="tf", suffix=0)

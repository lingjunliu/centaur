
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_mean_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "axis": np.array([0], dtype=np.int32),
        "keep_dims": False,
        "name": "mean_op1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        "axis": np.array([1], dtype=np.int32),
        "keep_dims": True,
        "name": "mean_op2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        "axis": np.array([0, 1], dtype=np.int32),
        "keep_dims": False,
        "name": "mean_op3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-1.0, -2.0, 3.0, 4.0, -5.0], dtype=np.float32),
        "axis": np.array([0], dtype=np.int32),
        "keep_dims": False,
        "name": "mean_op4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int64),
        "axis": np.array([-1], dtype=np.int32),
        "keep_dims": True,
        "name": "mean_op5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        "axis": np.array([0], dtype=np.int64),
        "keep_dims": False,
        "name": "mean_op6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int16),
        "axis": np.array([-2], dtype=np.int32),
        "keep_dims": True,
        "name": "mean_op7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[-10, 20, -30], [40, -50, 60]], dtype=np.int8),
        "axis": np.array([1], dtype=np.int64),
        "keep_dims": False,
        "name": "mean_op8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        "axis": np.array([0], dtype=np.int32),
        "keep_dims": True,
        "name": "mean_op9"
    }
    list_of

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Mean' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Mean'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Mean', generated_inputs['tf.raw_ops.Mean'], lib="tf", suffix=0)

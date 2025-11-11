
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_prod_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([1, 2, 3, 4], dtype=np.float32)
    axis = np.array(0, dtype=np.int32)
    keep_dims = False
    name = "prod_1d"
    input_dict = {
        "input": input_tensor,
        "axis": axis,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    axis = np.array(0, dtype=np.int32)
    keep_dims = True
    name = "prod_2d_axis0"
    input_dict = {
        "input": input_tensor,
        "axis": axis,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[2, 3], [4, 5]], dtype=np.int32)
    axis = np.array(1, dtype=np.int64)
    keep_dims = False
    name = "prod_2d_axis1"
    input_dict = {
        "input": input_tensor,
        "axis": axis,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = np.array(0, dtype=np.int32)
    keep_dims = True
    name = "prod_3d"
    input_dict = {
        "input": input_tensor,
        "axis": axis,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    axis = np.array(-1, dtype=np.int32)
    keep_dims = False
    name = "prod_negative_axis"
    input_dict = {
        "input": input_tensor,
        "axis": axis,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis = np.array([0, 1], dtype=np.int32)
    keep_dims = True
    name = "prod_multi_axis"
    input_dict = {
        "input": input_tensor,
        "axis": axis,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1, 2, 3, 4], dtype=np.uint8)
    axis = np.array(0, dtype=np.int32)
    keep_dims = False
    name = "prod_uint8"
    input_dict = {
        "input": input_tensor,
        "axis": axis,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int16)
    axis = np.array(0, dtype=np.int32)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Prod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Prod', generated_inputs['tf.raw_ops.Prod'], lib="tf", suffix=0)

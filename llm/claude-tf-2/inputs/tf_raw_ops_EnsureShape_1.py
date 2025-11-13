
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_ensureshape_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    shape = [5]
    name = "ensure_shape_1d"
    input_dict = {
        "input": input_tensor,
        "shape": shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    shape = [3, 2]
    name = "ensure_shape_2d"
    input_dict = {
        "input": input_tensor,
        "shape": shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    shape = [2, 2, 2]
    name = "ensure_shape_3d"
    input_dict = {
        "input": input_tensor,
        "shape": shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array(42, dtype=np.int64)
    shape = []
    name = "ensure_shape_scalar"
    input_dict = {
        "input": input_tensor,
        "shape": shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4, 5).astype(np.float32)
    shape = [2, 3, 4, 5]
    name = "ensure_shape_4d"
    input_dict = {
        "input": input_tensor,
        "shape": shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    shape = [-1, 3]
    name = "ensure_shape_partial"
    input_dict = {
        "input": input_tensor,
        "shape": shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-1, -2, -3, -4], dtype=np.int32)
    shape = [4]
    name = "ensure_shape_negative"
    input_dict = {
        "input": input_tensor,
        "shape": shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16)
    shape = [2, 2]
    name = "ensure_shape_float16"
    input_dict = {
        "input": input_tensor,
        "shape": shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.arange(100, dtype=np.int32)
    shape = [100]
    name = "ensure_shape_large"
    input_dict = {
        "input": input_tensor,
        "shape": shape,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.zeros((5, 10), dtype=np.float32)
    shape = [5, 10

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.EnsureShape_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EnsureShape_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.EnsureShape', generated_inputs['tf.raw_ops.EnsureShape_1'], lib="tf", suffix=1)

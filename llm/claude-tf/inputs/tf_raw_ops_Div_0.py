
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Div_inputs():
    list_of_inputs = []
    
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {"name": "div_op_1", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    y = np.array([2.0, 4.0], dtype=np.float32)
    input_dict = {"name": "div_op_2", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-10.0, 20.0, -30.0, 40.0], dtype=np.float64)
    y = np.array([2.0, -4.0, 5.0, -8.0], dtype=np.float64)
    input_dict = {"name": "div_op_3", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([100, 200, 300], dtype=np.int32)
    y = np.array([10, 20, 30], dtype=np.int32)
    input_dict = {"name": "div_op_4", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[[1.0, 1.0], [1.0, 2.0]], [[2.0, 3.0], [1.0, 4.0]]], dtype=np.float32)
    input_dict = {"name": "div_op_5", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1000], dtype=np.int64)
    y = np.array([10], dtype=np.int64)
    input_dict = {"name": "div_op_6", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    y = np.array([[[2.0]]], dtype=np.float32)
    input_dict = {"name": "div_op_7", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0+2.0j, 3.0+4.0j], dtype=np.complex64)
    y = np.array([1.0+1.0j, 2.0+0.0j], dtype=np.complex64)
    input_dict = {"name": "div_op_8", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([5.5, 10.5, 15.5], dtype=np.float32)
    y = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {"name": "div_op_9", "x": x, "y": y}
    list_of_

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Div' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Div'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Div', generated_inputs['tf.raw_ops.Div'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_pow_inputs():
    list_of_inputs = []
    
    x = np.array([[2, 2], [3, 3]], dtype=np.int32)
    y = np.array([[8, 16], [2, 3]], dtype=np.int32)
    input_dict = {"name": "pow_op_1", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(2.5, dtype=np.float32)
    y = np.array(3.0, dtype=np.float32)
    input_dict = {"name": "pow_op_2", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.array([2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    input_dict = {"name": "pow_op_3", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[-2, 3], [-4, 5]], dtype=np.int64)
    y = np.array([[2, 3], [2, 2]], dtype=np.int64)
    input_dict = {"name": "pow_op_4", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[[2.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [2.0, 2.0]]], dtype=np.float32)
    input_dict = {"name": "pow_op_5", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10, 20, 30], dtype=np.int16)
    y = np.array([0, 0, 0], dtype=np.int16)
    input_dict = {"name": "pow_op_6", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-2, -3, -4], dtype=np.int8)
    y = np.array([2, 2, 2], dtype=np.int8)
    input_dict = {"name": "pow_op_7", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[[2.0]]]], dtype=np.float32)
    y = np.array([[[[3.0]]]], dtype=np.float32)
    input_dict = {"name": "pow_op_8", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1+2j, 3+4j], dtype=np.complex64)
    y = np.array([2+0j, 1+0j], dtype=np.complex64)
    input_dict = {"name": "pow_op_9", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([5, 6, 7], dtype=np.int32)
    y = np.array([1, 1, 1], dtype=np.int

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Pow'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Pow', generated_inputs['tf.raw_ops.Pow'], lib="tf", suffix=0)

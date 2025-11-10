
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_raw_ops_greater_inputs():
    list_of_inputs = []
    
    input_dict = {
        "name": "greater_op_1",
        "x": np.array(5.0, dtype=np.float32),
        "y": np.array(3.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "greater_op_2",
        "x": np.array([5.0, 4.0, 6.0], dtype=np.float32),
        "y": np.array([5.0, 2.0, 5.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "greater_op_3",
        "x": np.array([5.0, 4.0, 6.0], dtype=np.float32),
        "y": np.array([5.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "greater_op_4",
        "x": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "y": np.array([[2, 1, 3], [3, 5, 7]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "greater_op_5",
        "x": np.array([-5, -3, 0, 3, 5], dtype=np.int64),
        "y": np.array([-4, -3, 0, 2, 6], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "greater_op_6",
        "x": np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float64),
        "y": np.array([[[1.0, 3.0], [3.0, 5.0]], [[5.0, 7.0], [7.0, 9.0]]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "greater_op_7",
        "x": np.array([10, 20, 30, 40], dtype=np.uint8),
        "y": np.array([15, 15, 30, 35], dtype=np.uint8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "greater_op_8",
        "x": np.array([-128, -50, 0, 50, 127], dtype=np.int8),
        "y": np.array([-100, -50, 1, 40, 120], dtype=np.int8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "greater_op_9",
        "x": np.array([1000, 2000, 3000], dtype=np.int16),
        "y": np.array([1500, 1500, 3000], dtype=np.int16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "greater_op_10",
        "x": np.array([100, 200,

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Greater' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Greater'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Greater', generated_inputs['tf.raw_ops.Greater'], lib="tf", suffix=0)

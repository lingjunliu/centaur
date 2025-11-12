
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DiagPart_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([[1, 0, 0, 0],
                            [0, 2, 0, 0],
                            [0, 0, 3, 0],
                            [0, 0, 0, 4]], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "name": "diag_part_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[-1, 2],
                            [3, -4]], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "diag_part_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[[1, 2], [3, 4]],
                             [[5, 6], [7, 8]]],
                            [[[9, 10], [11, 12]],
                             [[13, 14], [15, 16]]]], dtype=np.float64)
    input_dict = {
        "input": input_tensor,
        "name": "diag_part_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.5, 2.5, 3.5],
                            [4.5, 5.5, 6.5],
                            [7.5, 8.5, 9.5]], dtype=np.float64)
    input_dict = {
        "input": input_tensor,
        "name": "diag_part_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[42]], dtype=np.int64)
    input_dict = {
        "input": input_tensor,
        "name": "diag_part_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(3, 2, 3, 2).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "name": "diag_part_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.eye(5, dtype=np.int64) * np.arange(1, 6)
    input_dict = {
        "input": input_tensor,
        "name": "diag_part_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.ones((2, 2, 2, 2, 2, 2), dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "name": "diag_part_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1+2j, 3+4j],
                            [5+6j, 7+8j]], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "name": "diag_part_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.zeros((4, 4), dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "name": "diag_part_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DiagPart' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DiagPart'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DiagPart', generated_inputs['tf.raw_ops.DiagPart'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_linalg_lstsq_inputs():
    list_of_inputs = []
    
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    rhs = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": 0.0,
        "fast": True,
        "name": "lstsq_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    rhs = np.array([[1.0], [2.0]], dtype=np.float32)
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": 0.0,
        "fast": True,
        "name": "lstsq_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": 0.5,
        "fast": True,
        "name": "lstsq_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    matrix = np.array([[2.0, 1.0], [1.0, 2.0], [1.0, 1.0]], dtype=np.float32)
    rhs = np.array([[3.0], [4.0], [2.0]], dtype=np.float32)
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": 0.0,
        "fast": False,
        "name": "lstsq_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    matrix = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]], dtype=np.float32)
    rhs = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], dtype=np.float32)
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": 0.0,
        "fast": True,
        "name": "lstsq_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    matrix = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    rhs = np.array([[5.0], [6.0], [7.0], [8.0]], dtype=np.float32)
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "l2_regularizer": 0.1,
        "fast": True,
        "name": "lstsq_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    matrix = np

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.lstsq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.lstsq'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.lstsq', generated_inputs['tf.linalg.lstsq'], lib="tf", suffix=0)

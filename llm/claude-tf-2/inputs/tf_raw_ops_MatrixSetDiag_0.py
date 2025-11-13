
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixSetDiag_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
    diagonal = np.array([10, 20], dtype=np.float32)
    input_dict = {
        "name": "matrix_set_diag_1",
        "input": input_tensor,
        "diagonal": diagonal
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    diagonal = np.array([-1, -2, -3], dtype=np.float32)
    input_dict = {
        "name": "matrix_set_diag_2",
        "input": input_tensor,
        "diagonal": diagonal
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    diagonal = np.array([100, 200], dtype=np.float32)
    input_dict = {
        "name": "matrix_set_diag_3",
        "input": input_tensor,
        "diagonal": diagonal
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    diagonal = np.array([10, 20], dtype=np.float32)
    input_dict = {
        "name": "matrix_set_diag_4",
        "input": input_tensor,
        "diagonal": diagonal
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    diagonal = np.array([[10, 20], [30, 40]], dtype=np.float32)
    input_dict = {
        "name": "matrix_set_diag_5",
        "input": input_tensor,
        "diagonal": diagonal
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]], dtype=np.float32)
    diagonal = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.float32)
    input_dict = {
        "name": "matrix_set_diag_6",
        "input": input_tensor,
        "diagonal": diagonal
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    diagonal = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {
        "name": "matrix_set_diag_7",
        "input": input_tensor,
        "diagonal": diagonal
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    diagonal = np.array([9.9, 8.8], dtype=np.float64)
    input_dict = {
        "name": "matrix_set_diag_8",
        "input

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MatrixSetDiag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixSetDiag'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MatrixSetDiag', generated_inputs['tf.raw_ops.MatrixSetDiag'], lib="tf", suffix=0)

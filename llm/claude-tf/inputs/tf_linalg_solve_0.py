
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_linalg_solve_inputs():
    list_of_inputs = []
    
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    rhs = np.array([[9.0], [8.0]], dtype=np.float32)
    adjoint = False
    name = "solve1"
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    matrix = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]], dtype=np.float64)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    adjoint = False
    name = "solve2"
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    matrix = np.array([[4.0, 2.0], [1.0, 3.0]], dtype=np.float32)
    rhs = np.array([[10.0], [7.0]], dtype=np.float32)
    adjoint = True
    name = "solve3"
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    matrix = np.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 3.0]]], dtype=np.float32)
    rhs = np.array([[[5.0], [6.0]], [[8.0], [12.0]]], dtype=np.float32)
    adjoint = False
    name = "solve4"
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    matrix = np.array([[1.0+1.0j, 0.5], [0.5, 1.0-1.0j]], dtype=np.complex64)
    rhs = np.array([[2.0+1.0j], [1.0-1.0j]], dtype=np.complex64)
    adjoint = False
    name = "solve5"
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    matrix = np.array([[2.0+0.5j, 1.0], [1.0, 2.0-0.5j]], dtype=np.complex128)
    rhs = np.array([[3.0+2.0j], [4.0-1.0j]], dtype=np.complex128)
    adjoint = True
    name = "solve6"
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    matrix = np.array([[5.0, -2.0], [-2.0, 4.0]], dtype=np.float32)
    rhs = np.array([[1.0, 2.0, 3.0

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.solve'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.solve', generated_inputs['tf.linalg.solve'], lib="tf", suffix=0)

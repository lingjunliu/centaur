
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_linalg_solve_inputs():
    list_of_inputs = []
    
    # Input 1: float32 matrix, float32 rhs
    matrix = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    adjoint = False
    name = "solve_1"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: float64 matrix, float64 rhs
    matrix = np.array([[3.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    adjoint = True
    name = "solve_2"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: complex64 matrix, complex64 rhs
    matrix = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    rhs = np.array([[9+10j, 11+12j], [13+14j, 15+16j]], dtype=np.complex64)
    adjoint = False
    name = "solve_3"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: complex128 matrix, complex128 rhs
    matrix = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    rhs = np.array([[9+10j, 11+12j], [13+14j, 15+16j]], dtype=np.complex128)
    adjoint = True
    name = "solve_4"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: 2D tensor matrix, 2D tensor rhs
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    adjoint = False
    name = "solve_5"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: 2D tensor matrix, 2D tensor rhs with adjoint=True
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    adjoint = True
    name = "solve_6"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: negative values matrix, negative values rhs
    matrix = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    rhs = np.array([[-5.0, -6.0], [-7.0, -8.0]], dtype=np.float64)
    adjoint = False
    name = "solve_7"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: float32 matrix, float32 rhs with different dimensions (valid matrix)
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    adjoint = False
    name = "solve_8"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: float64 matrix, float64 rhs with different dimensions and adjoint=True (valid matrix)
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    adjoint = True
    name = "solve_9"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: half matrix, half rhs
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float16)
    adjoint = False
    name = "solve_10"
    
    input_dict = {
        "matrix": matrix,
        "rhs": rhs,
        "adjoint": adjoint,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.linalg.solve"] = tf_linalg_solve_inputs()

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


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_lu_solve_inputs():
    list_of_inputs = []

    # Input 1
    lower_upper = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    perm = np.array([1, 0], dtype=np.int32)
    rhs = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    validate_args = False
    name = "lu_solve_1"

    input_dict = {
        "lower_upper": tf.convert_to_tensor(lower_upper),
        "perm": tf.convert_to_tensor(perm),
        "rhs": tf.convert_to_tensor(rhs),
        "validate_args": validate_args,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    lower_upper = np.array([[7., 8.], [3., 4.]], dtype=np.float64)
    perm = np.array([0, 1], dtype=np.int32)
    rhs = np.array([[1., 2.], [3., 4.]], dtype=np.float64)
    validate_args = True
    name = "lu_solve_2"

    input_dict = {
        "lower_upper": tf.convert_to_tensor(lower_upper),
        "perm": tf.convert_to_tensor(perm),
        "rhs": tf.convert_to_tensor(rhs),
        "validate_args": validate_args,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 (3D)
    lower_upper = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32)
    perm = np.array([[1, 0], [0, 1]], dtype=np.int32)
    rhs = np.array([[[1., 0.], [0., 1.]], [[1., 1.], [0., 0.]]], dtype=np.float32)
    validate_args = False
    name = "lu_solve_3"

    input_dict = {
        "lower_upper": tf.convert_to_tensor(lower_upper),
        "perm": tf.convert_to_tensor(perm),
        "rhs": tf.convert_to_tensor(rhs),
        "validate_args": validate_args,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (different shapes)
    lower_upper = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
    perm = np.array([2, 0, 1], dtype=np.int32)
    rhs = np.array([[1.], [0.], [1.]], dtype=np.float32)
    validate_args = True
    name = "lu_solve_4"

    input_dict = {
        "lower_upper": tf.convert_to_tensor(lower_upper),
        "perm": tf.convert_to_tensor(perm),
        "rhs": tf.convert_to_tensor(rhs),
        "validate_args": validate_args,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    lower_upper = np.array([[1.j, 2.], [3., 4.]], dtype=np.complex64)
    perm = np.array([1, 0], dtype=np.int32)
    rhs = np.array([[1., 0.], [0., 1.j]], dtype=np.complex64)
    validate_args = False
    name = "lu_solve_5"

    input_dict = {
        "lower_upper": tf.convert_to_tensor(lower_upper),
        "perm": tf.convert_to_tensor(perm),
        "rhs": tf.convert_to_tensor(rhs),
        "validate_args": validate_args,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (negative values)
    lower_upper = np.array([[-1., 2.], [3., -4.]], dtype=np.float32)
    perm = np.array([1, 0], dtype=np.int32)
    rhs = np.array([[1., 0.], [0., -1.]], dtype=np.float32)
    validate_args = False
    name = "lu_solve_6"

    input_dict = {
        "lower_upper": tf.convert_to_tensor(lower_upper),
        "perm": tf.convert_to_tensor(perm),
        "rhs": tf.convert_to_tensor(rhs),
        "validate_args": validate_args,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (scalar rhs)
    lower_upper = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    perm = np.array([1, 0], dtype=np.int32)
    rhs = np.array([[1.], [0.]], dtype=np.float32)
    validate_args = True
    name = "lu_solve_7"

    input_dict = {
        "lower_upper": tf.convert_to_tensor(lower_upper),
        "perm": tf.convert_to_tensor(perm),
        "rhs": tf.convert_to_tensor(rhs),
        "validate_args": validate_args,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (batch and complex)
    lower_upper = np.array([[[1. + 1.j, 2.], [3., 4.]], [[5., 6.], [7., 8. + 1.j]]], dtype=np.complex64)
    perm = np.array([[1, 0], [0, 1]], dtype=np.int32)
    rhs = np.array([[[1., 0.], [0., 1.]], [[1., 1.], [0., 0.]]], dtype=np.float32).astype(np.complex64)
    validate_args = False
    name = "lu_solve_8"

    input_dict = {
        "lower_upper": tf.convert_to_tensor(lower_upper),
        "perm": tf.convert_to_tensor(perm),
        "rhs": tf.convert_to_tensor(rhs),
        "validate_args": validate_args,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (3x3 matrices)
    lower_upper = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
    perm = np.array([2, 0, 1], dtype=np.int32)
    rhs = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)
    validate_args = True
    name = "lu_solve_9"

    input_dict = {
        "lower_upper": tf.convert_to_tensor(lower_upper),
        "perm": tf.convert_to_tensor(perm),
        "rhs": tf.convert_to_tensor(rhs),
        "validate_args": validate_args,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (Batch 3x3 matrices)
    lower_upper = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                            [[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]], dtype=np.float32)
    perm = np.array([[2, 0, 1], [0, 1, 2]], dtype=np.int32)
    rhs = np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], dtype=np.float32)
    validate_args = False
    name = "lu_solve_10"

    input_dict = {
        "lower_upper": tf.convert_to_tensor(lower_upper),
        "perm": tf.convert_to_tensor(perm),
        "rhs": tf.convert_to_tensor(rhs),
        "validate_args": validate_args,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.lu_solve"] = tf_linalg_lu_solve_inputs()
for k, v in generated_inputs.items():
    for i, d in enumerate(v):
        for k1, v1 in d.items():
            if isinstance(v1, tf.Tensor):
                d[k1] = v1.numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.lu_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.lu_solve'.")

check_valid('tf.linalg.lu_solve', generated_inputs['tf.linalg.lu_solve'], lib="tf", suffix=0)

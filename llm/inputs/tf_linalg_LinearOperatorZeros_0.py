
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorzeros_inputs():
    list_of_inputs = []

    # Input 1
    num_rows = np.int32(2)
    num_columns = np.int32(2)
    batch_shape = None
    dtype = tf.float32
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = False
    name = "zeros_op_1"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": [] if batch_shape is None else batch_shape,
        "dtype": dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "assert_proper_shapes": assert_proper_shapes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_rows = np.int32(5)
    num_columns = None
    batch_shape = [np.int32(2), np.int32(3)]
    dtype = tf.complex64
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = False
    name = "zeros_op_2"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_rows,
        "batch_shape": list(batch_shape) if batch_shape is not None else [],
        "dtype": dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "assert_proper_shapes": assert_proper_shapes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_rows = np.int32(1)
    num_columns = np.int32(1)
    batch_shape = [np.int32(4)]
    dtype = tf.float64
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = True
    name = "zeros_op_3"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": list(batch_shape) if batch_shape is not None else [],
        "dtype": dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": assert_proper_shapes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    num_rows = np.int32(10)
    num_columns = None
    batch_shape = None
    dtype = tf.int32
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = False
    name = "zeros_op_4"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_rows,
        "batch_shape": [] if batch_shape is None else batch_shape,
        "dtype": dtype,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": assert_proper_shapes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    num_rows = np.int32(7)
    num_columns = np.int32(7)
    batch_shape = [np.int32(1), np.int32(5)]
    dtype = tf.complex128
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = True
    name = "zeros_op_5"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": list(batch_shape) if batch_shape is not None else [],
        "dtype": dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": assert_proper_shapes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    num_rows = np.int32(3)
    num_columns = np.int32(3)
    batch_shape = None
    dtype = tf.bfloat16
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = False
    name = "zeros_op_6"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": [] if batch_shape is None else batch_shape,
        "dtype": dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": assert_proper_shapes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    num_rows = np.int32(4)
    num_columns = None
    batch_shape = [np.int32(2)]
    dtype = tf.float16
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = False
    name = "zeros_op_7"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_rows,
        "batch_shape": list(batch_shape) if batch_shape is not None else [],
        "dtype": dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": assert_proper_shapes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    num_rows = np.int32(6)
    num_columns = np.int32(1)
    batch_shape = [np.int32(3), np.int32(2), np.int32(1)]
    dtype = tf.float32
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = False
    assert_proper_shapes = False
    name = "zeros_op_8"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": list(batch_shape) if batch_shape is not None else [],
        "dtype": dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": assert_proper_shapes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    num_rows = np.int32(8)
    num_columns = None
    batch_shape = None
    dtype = tf.complex64
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = False
    name = "zeros_op_9"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_rows,
        "batch_shape": [] if batch_shape is None else batch_shape,
        "dtype": dtype,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": assert_proper_shapes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    num_rows = np.int32(9)
    num_columns = np.int32(9)
    batch_shape = [np.int32(5)]
    dtype = tf.int64
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = True
    name = "zeros_op_10"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "batch_shape": list(batch_shape) if batch_shape is not None else [],
        "dtype": dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": assert_proper_shapes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    num_rows = np.int32(2)
    num_columns = None
    batch_shape = [np.int32(1), np.int32(2), np.int32(3)]
    dtype = tf.float32
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    assert_proper_shapes = False
    name = "zeros_op_11"

    input_dict = {
        "num_rows": num_rows,
        "num_columns": num_rows,
        "batch_shape": list(batch_shape) if batch_shape is not None else [],
        "dtype": dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": assert_proper_shapes,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorZeros"] = tf_linalg_linearoperatorzeros_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorZeros' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorZeros'.")

check_valid('tf.linalg.LinearOperatorZeros', generated_inputs['tf.linalg.LinearOperatorZeros'], lib="tf", suffix=0)

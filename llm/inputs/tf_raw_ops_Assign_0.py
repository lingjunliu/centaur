
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_assign_inputs():
    list_of_inputs = []

    # Input 1: Basic assignment
    ref_np = np.array([1, 2, 3], dtype=np.int32)
    ref = tf.Variable(ref_np)
    value = tf.constant(np.array([4, 5, 6], dtype=np.int32))
    validate_shape = True
    use_locking = True
    name = "assign_op_1"

    input_dict = {
        "ref": ref.numpy(),
        "value": value.numpy(),
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Assignment with shape validation disabled
    ref_np = np.array([1, 2, 3], dtype=np.int32)
    ref = tf.Variable(ref_np)
    value = tf.constant(np.array([4, 5, 6], dtype=np.int32))
    validate_shape = False
    use_locking = False
    name = "assign_op_2"

    input_dict = {
        "ref": ref.numpy(),
        "value": value.numpy(),
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional assignment
    ref_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    ref = tf.Variable(ref_np)
    value = tf.constant(np.array([[5, 6], [7, 8]], dtype=np.float32))
    validate_shape = True
    use_locking = True
    name = "assign_op_3"

    input_dict = {
        "ref": ref.numpy(),
        "value": value.numpy(),
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Assignment with different data type
    ref_np = np.array([1, 2, 3], dtype=np.int64)
    ref = tf.Variable(ref_np)
    value = tf.constant(np.array([4, 5, 6], dtype=np.int64))
    validate_shape = True
    use_locking = True
    name = "assign_op_4"

    input_dict = {
        "ref": ref.numpy(),
        "value": value.numpy(),
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Assignment with a different name
    ref_np = np.array([1, 2, 3], dtype=np.int32)
    ref = tf.Variable(ref_np)
    value = tf.constant(np.array([7, 8, 9], dtype=np.int32))
    validate_shape = True
    use_locking = True
    name = "different_name"

    input_dict = {
        "ref": ref.numpy(),
        "value": value.numpy(),
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float data type
    ref_np = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    ref = tf.Variable(ref_np)
    value = tf.constant(np.array([4.5, 5.5, 6.5], dtype=np.float32))
    validate_shape = True
    use_locking = True
    name = "float_assign"

    input_dict = {
        "ref": ref.numpy(),
        "value": value.numpy(),
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float
    ref_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    ref = tf.Variable(ref_np)
    value = tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64))
    validate_shape = True
    use_locking = True
    name = "2d_float"

    input_dict = {
        "ref": ref.numpy(),
        "value": value.numpy(),
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boolean values
    ref_np = np.array([True, False, True], dtype=np.bool_)
    ref = tf.Variable(ref_np)
    value = tf.constant(np.array([False, True, False], dtype=np.bool_))
    validate_shape = True
    use_locking = True
    name = "bool_assign"

    input_dict = {
        "ref": ref.numpy(),
        "value": value.numpy(),
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Int8
    ref_np = np.array([1, 2, 3], dtype=np.int8)
    ref = tf.Variable(ref_np)
    value = tf.constant(np.array([4, 5, 6], dtype=np.int8))
    validate_shape = True
    use_locking = True
    name = "int8_assign"

    input_dict = {
        "ref": ref.numpy(),
        "value": value.numpy(),
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Rank 3 Tensor
    ref_np = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    ref = tf.Variable(ref_np)
    value = tf.constant(np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32))
    validate_shape = True
    use_locking = True
    name = "rank_3_assign"

    input_dict = {
        "ref": ref.numpy(),
        "value": value.numpy(),
        "validate_shape": validate_shape,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Assign"] = tf_raw_ops_assign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Assign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Assign'.")

check_valid('tf.raw_ops.Assign', generated_inputs['tf.raw_ops.Assign'], lib="tf", suffix=0)

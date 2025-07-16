
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_assign_sub_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 subtraction with locking
    ref = tf.Variable(np.array(5.0, dtype=np.float32))
    value = tf.constant(np.array(2.0, dtype=np.float32))
    use_locking = True
    name = "assign_sub_float32_locking"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic int32 subtraction without locking
    ref = tf.Variable(np.array(10, dtype=np.int32))
    value = tf.constant(np.array(3, dtype=np.int32))
    use_locking = False
    name = "assign_sub_int32_no_locking"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multidimensional float64 subtraction
    ref = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    value = tf.constant(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64))
    use_locking = False
    name = "assign_sub_float64_multi"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64 subtraction with negative value
    ref = tf.Variable(np.array(20, dtype=np.int64))
    value = tf.constant(np.array(-5, dtype=np.int64))
    use_locking = True
    name = "assign_sub_int64_negative"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 subtraction
    ref = tf.Variable(np.array(complex(2, 3), dtype=np.complex64))
    value = tf.constant(np.array(complex(1, 1), dtype=np.complex64))
    use_locking = False
    name = "assign_sub_complex64"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multidimensional int32 subtraction with different shapes
    ref = tf.Variable(np.array([1, 2, 3, 4, 5, 6], dtype=np.int32).reshape((2,3)))
    value = tf.constant(np.array([1, 1, 1, 1, 1, 1], dtype=np.int32).reshape((2,3)))
    use_locking = True
    name = "assign_sub_int32_multi_shape"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8 subtraction (ensure positive result)
    ref = tf.Variable(np.array(250, dtype=np.uint8))
    value = tf.constant(np.array(5, dtype=np.uint8))
    use_locking = True
    name = "assign_sub_uint8"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: qint32 subtraction. Using int32 as a substitute.
    ref = tf.Variable(np.array(100, dtype=np.int32))
    value = tf.constant(np.array(20, dtype=np.int32))
    use_locking = False
    name = "assign_sub_qint32"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: half subtraction. Using float16 as a substitute.
    ref = tf.Variable(np.array(5.0, dtype=np.float16))
    value = tf.constant(np.array(2.0, dtype=np.float16))
    use_locking = False
    name = "assign_sub_half"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scalar float32
    ref = tf.Variable(np.array(7.0, dtype=np.float32))
    value = tf.constant(np.array(1.0, dtype=np.float32))
    use_locking = False
    name = "assign_sub_scalar_float32"

    input_dict = {
        "ref": ref,
        "value": value,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return [
        {"ref": tf.Variable(np.array(5.0, dtype=np.float32)), "value": tf.constant(np.array(2.0, dtype=np.float32)), "use_locking": True, "name": "assign_sub_float32_locking"},
        {"ref": tf.Variable(np.array(10, dtype=np.int32)), "value": tf.constant(np.array(3, dtype=np.int32)), "use_locking": False, "name": "assign_sub_int32_no_locking"},
        {"ref": tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)), "value": tf.constant(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)), "use_locking": False, "name": "assign_sub_float64_multi"},
        {"ref": tf.Variable(np.array(20, dtype=np.int64)), "value": tf.constant(np.array(-5, dtype=np.int64)), "use_locking": True, "name": "assign_sub_int64_negative"},
        {"ref": tf.Variable(np.array(complex(2, 3), dtype=np.complex64)), "value": tf.constant(np.array(complex(1, 1), dtype=np.complex64)), "use_locking": False, "name": "assign_sub_complex64"},
        {"ref": tf.Variable(np.array([1, 2, 3, 4, 5, 6], dtype=np.int32).reshape((2,3))), "value": tf.constant(np.array([1, 1, 1, 1, 1, 1], dtype=np.int32).reshape((2,3))), "use_locking": True, "name": "assign_sub_int32_multi_shape"},
        {"ref": tf.Variable(np.array(250, dtype=np.uint8)), "value": tf.constant(np.array(5, dtype=np.uint8)), "use_locking": True, "name": "assign_sub_uint8"},
        {"ref": tf.Variable(np.array(100, dtype=np.int32)), "value": tf.constant(np.array(20, dtype=np.int32)), "use_locking": False, "name": "assign_sub_qint32"},
        {"ref": tf.Variable(np.array(5.0, dtype=np.float16)), "value": tf.constant(np.array(2.0, dtype=np.float16)), "use_locking": False, "name": "assign_sub_half"},
        {"ref": tf.Variable(np.array(7.0, dtype=np.float32)), "value": tf.constant(np.array(1.0, dtype=np.float32)), "use_locking": False, "name": "assign_sub_scalar_float32"}
    ]

generated_inputs = {}
generated_inputs["tf.raw_ops.AssignSub"] = tf_raw_ops_assign_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.AssignSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.AssignSub'.")

check_valid('tf.raw_ops.AssignSub', generated_inputs['tf.raw_ops.AssignSub'], lib="tf", suffix=0)

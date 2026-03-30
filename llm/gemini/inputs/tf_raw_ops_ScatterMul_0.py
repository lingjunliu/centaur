
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_ScatterMul_inputs():
    list_of_inputs = []

    # Input 1
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2.0, 3.0], dtype=np.float32)
    use_locking = False
    name = "scatter_mul_1"

    ref_var = tf.Variable(ref)

    input_dict = {
        "ref": ref_var,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([[2, 2], [3, 3]], dtype=np.int32)
    use_locking = True
    name = "scatter_mul_2"

    ref_var = tf.Variable(ref)


    input_dict = {
        "ref": ref_var,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    indices = np.array([1], dtype=np.int64)
    updates = np.array([5.0], dtype=np.float64)
    use_locking = False
    name = "scatter_mul_3"

    ref_var = tf.Variable(ref)

    input_dict = {
        "ref": ref_var,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ref = np.array([1, 2, 3, 4], dtype=np.int64)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    updates = np.array([2, 3, 4, 5], dtype=np.int64)
    use_locking = True
    name = "scatter_mul_4"

    ref_var = tf.Variable(ref)

    input_dict = {
        "ref": ref_var,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ref = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([[2, 3], [4, 5]], dtype=np.float32)
    use_locking = False
    name = "scatter_mul_5"
    ref_var = tf.Variable(ref)

    input_dict = {
        "ref": ref_var,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ref = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    updates = np.array([2, 3, 4], dtype=np.int32)
    use_locking = True
    name = "scatter_mul_6"

    ref_var = tf.Variable(ref)


    input_dict = {
        "ref": ref_var,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([3, 1], dtype=np.int32)
    updates = np.array([2.0, 3.0], dtype=np.float32)
    use_locking = False
    name = "scatter_mul_7"

    ref_var = tf.Variable(ref)

    input_dict = {
        "ref": ref_var,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([2, 1], dtype=np.int32)
    updates = np.array([[3, 3], [2, 2]], dtype=np.int32)
    use_locking = True
    name = "scatter_mul_8"

    ref_var = tf.Variable(ref)

    input_dict = {
        "ref": ref_var,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    indices = np.array([0], dtype=np.int64)
    updates = np.array([0.5], dtype=np.float64)
    use_locking = False
    name = "scatter_mul_9"

    ref_var = tf.Variable(ref)

    input_dict = {
        "ref": ref_var,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ref = np.array([1, 2, 3, 4], dtype=np.int64)
    indices = np.array([2, 0], dtype=np.int32)
    updates = np.array([5, 2], dtype=np.int64)
    use_locking = True
    name = "scatter_mul_10"

    ref_var = tf.Variable(ref)

    input_dict = {
        "ref": ref_var,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterMul"] = tf_raw_ops_ScatterMul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMul'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ScatterMul', generated_inputs['tf.raw_ops.ScatterMul'], lib="tf", suffix=0)

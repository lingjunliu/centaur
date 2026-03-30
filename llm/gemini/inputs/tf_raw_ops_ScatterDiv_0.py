
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_scatterdiv_inputs():
    list_of_inputs = []

    # Input 1: Simple case with float32
    ref = tf.Variable(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)).ref()
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2.0, 3.0], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32
    ref = tf.Variable(np.array([1, 2, 3, 4], dtype=np.int32)).ref()
    indices = np.array([1, 3], dtype=np.int32)
    updates = np.array([2, 2], dtype=np.int32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": True, "name": "int32_case"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional ref
    ref = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)).ref()
    indices = np.array([0, 2], dtype=np.int64)
    updates = np.array([[2.0, 2.0], [5.0, 3.0]], dtype=np.float64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different index order
    ref = tf.Variable(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)).ref()
    indices = np.array([3, 1], dtype=np.int32)
    updates = np.array([2.0, 3.0], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero updates
    ref = tf.Variable(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)).ref()
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64 type
    ref = tf.Variable(np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex64)).ref()
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2+0j, 3+1j], dtype=np.complex64)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Higher rank indices
    ref = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)).ref()
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.array([[[2.0, 1.0]],[[3.0, 2.0]]], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint8
    ref = tf.Variable(np.array([1, 2, 3, 4], dtype=np.uint8)).ref()
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2, 3], dtype=np.uint8)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty updates
    ref = tf.Variable(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)).ref()
    indices = np.array([], dtype=np.int32)
    updates = np.array([], dtype=np.float32)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128 type
    ref = tf.Variable(np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex128)).ref()
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2+0j, 3+1j], dtype=np.complex128)
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterDiv"] = tf_raw_ops_scatterdiv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterDiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterDiv'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ScatterDiv', generated_inputs['tf.raw_ops.ScatterDiv'], lib="tf", suffix=0)

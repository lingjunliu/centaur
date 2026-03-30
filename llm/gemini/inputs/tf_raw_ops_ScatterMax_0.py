
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_scatter_max_inputs():
    list_of_inputs = []

    # Input 1: Simple case with float32
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([5.0, 4.0], dtype=np.float32)
    input_dict = {"ref": tf.Variable(ref), "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int64 with use_locking
    ref = np.array([1, 2, 3, 4], dtype=np.int64)
    indices = np.array([1, 3], dtype=np.int64)
    updates = np.array([5, 6], dtype=np.int64)
    input_dict = {"ref": tf.Variable(ref), "indices": indices, "updates": updates, "use_locking": True, "name": "scatter_max"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional ref
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([[7, 8], [9, 10]], dtype=np.float32)
    input_dict = {"ref": tf.Variable(ref), "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar updates
    ref = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array([1], dtype=np.int32)
    updates = np.array(5, dtype=np.int32)
    input_dict = {"ref": tf.Variable(ref), "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different index type
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int64)
    updates = np.array([5.0, 4.0], dtype=np.float32)
    input_dict = {"ref": tf.Variable(ref), "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float16 type
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([5.0, 4.0], dtype=np.float16)
    input_dict = {"ref": tf.Variable(ref), "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: negative values
    ref = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([-5.0, 4.0], dtype=np.float32)
    input_dict = {"ref": tf.Variable(ref), "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: higher rank indices
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    indices = np.array([[0], [1]], dtype=np.int32)
    updates = np.array([[7,8], [9,10]], dtype=np.float32)
    input_dict = {"ref": tf.Variable(ref), "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([5.0, 4.0], dtype=np.float64)
    input_dict = {"ref": tf.Variable(ref), "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int32
    ref = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([5, 4], dtype=np.int32)
    input_dict = {"ref": tf.Variable(ref), "indices": indices, "updates": updates, "use_locking": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterMax"] = tf_raw_ops_scatter_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterMax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMax'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ScatterMax', generated_inputs['tf.raw_ops.ScatterMax'], lib="tf", suffix=0)

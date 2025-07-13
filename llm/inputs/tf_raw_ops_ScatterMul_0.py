
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ScatterMul_inputs():
    list_of_inputs = []

    # Input 1: Simple case with scalar indices
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([2.0, 3.0], dtype=np.float32)
    use_locking = False
    name = None
    input_dict = {"ref": tf.Variable(ref), "indices": tf.convert_to_tensor(indices), "updates": tf.convert_to_tensor(updates), "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Vector indices
    ref = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    use_locking = True
    name = "scatter_mul_1"
    input_dict = {"ref": tf.Variable(ref), "indices": tf.convert_to_tensor(indices), "updates": tf.convert_to_tensor(updates), "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer type
    ref = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array([1], dtype=np.int32)
    updates = np.array([5], dtype=np.int32)
    use_locking = False
    name = "scatter_mul_2"
    input_dict = {"ref": tf.Variable(ref), "indices": tf.convert_to_tensor(indices), "updates": tf.convert_to_tensor(updates), "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float64 type
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    indices = np.array([0, 1, 2], dtype=np.int64)
    updates = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    use_locking = True
    name = "scatter_mul_3"
    input_dict = {"ref": tf.Variable(ref), "indices": tf.convert_to_tensor(indices), "updates": tf.convert_to_tensor(updates), "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional updates
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([[2, 3], [4, 5]], dtype=np.int32)
    use_locking = False
    name = "scatter_mul_4"
    input_dict = {"ref": tf.Variable(ref), "indices": tf.convert_to_tensor(indices), "updates": tf.convert_to_tensor(updates), "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty updates
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([], dtype=np.int32)
    updates = np.array([], dtype=np.float32)
    use_locking = True
    name = "scatter_mul_5"
    input_dict = {"ref": tf.Variable(ref), "indices": tf.convert_to_tensor(indices), "updates": tf.convert_to_tensor(updates), "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Int64 indices
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int64)
    updates = np.array([2.0, 3.0], dtype=np.float32)
    use_locking = False
    name = None
    input_dict = {"ref": tf.Variable(ref), "indices": tf.convert_to_tensor(indices), "updates": tf.convert_to_tensor(updates), "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Rank 3 ref and updates
    ref = np.ones((2, 3, 4), dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.ones((2, 3, 4), dtype=np.float32) * 2
    use_locking = False
    name = None
    input_dict = {"ref": tf.Variable(ref), "indices": tf.convert_to_tensor(indices), "updates": tf.convert_to_tensor(updates), "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8
    ref = np.array([1, 2, 3], dtype=np.uint8)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([2, 3], dtype=np.uint8)
    use_locking = False
    name = None
    input_dict = {"ref": tf.Variable(ref), "indices": tf.convert_to_tensor(indices), "updates": tf.convert_to_tensor(updates), "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float16
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([2.0, 3.0], dtype=np.float16)
    use_locking = False
    name = None
    input_dict = {"ref": tf.Variable(ref), "indices": tf.convert_to_tensor(indices), "updates": tf.convert_to_tensor(updates), "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterMul"] = tf_raw_ops_ScatterMul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMul'.")

check_valid('tf.raw_ops.ScatterMul', generated_inputs['tf.raw_ops.ScatterMul'], lib="tf", suffix=0)

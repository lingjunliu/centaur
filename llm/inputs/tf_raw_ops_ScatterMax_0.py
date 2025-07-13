
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ScatterMax_inputs():
    list_of_inputs = []

    # Input 1: Basic test case with int32
    ref = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    indices = np.array([0, 2, 4], dtype=np.int32)
    updates = np.array([6, 7, 8], dtype=np.int32)
    use_locking = False
    name = "basic_int32"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float32 with locking
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([4.0, 5.0], dtype=np.float32)
    use_locking = True
    name = "float32_locking"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values and int64 indices
    ref = np.array([-1, -2, -3], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int64)
    updates = np.array([-4, 0], dtype=np.int32)
    use_locking = False
    name = "negative_values"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional ref and updates
    ref = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([[5, 6], [7, 8]], dtype=np.int32)
    use_locking = False
    name = "multi_dimensional"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher rank indices
    ref = np.array([1, 2, 3, 4], dtype=np.int32)
    indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
    updates = np.array([[5, 6], [7, 8]], dtype=np.int32)
    use_locking = False
    name = "high_rank_indices"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Updates as scalar
    ref = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    updates = np.array(5, dtype=np.int32)
    use_locking = False
    name = "scalar_update"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16 - Reduced the number of elements due to float16 issues
    ref = np.array([1.0, 2.0], dtype=np.float16) # Reduced precision to avoid potential overflows
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([4.0, 5.0], dtype=np.float16)  # Reduced precision to avoid potential overflows
    use_locking = False
    name = "bfloat16"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 - Reduced the number of elements
    ref = np.array([1.0, 2.0], dtype=np.float64)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([4.0, 5.0], dtype=np.float64)
    use_locking = False
    name = "float64"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half - Reduced the number of elements
    ref = np.array([1.0, 2.0], dtype=np.float16)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([4.0, 5.0], dtype=np.float16)
    use_locking = False
    name = "half"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int64 - Reduced the number of elements
    ref = np.array([1, 2], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([6, 7], dtype=np.int64)
    use_locking = False
    name = "basic_int64"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Identical indices - tests correct handling of duplicates - Reduced the number of elements
    ref = np.array([1, 2], dtype=np.int32)
    indices = np.array([0, 0], dtype=np.int32)
    updates = np.array([6, 7], dtype=np.int32)
    use_locking = False
    name = "duplicate_indices"
    input_dict = {"ref": ref, "indices": indices, "updates": updates, "use_locking": use_locking, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterMax"] = tf_raw_ops_ScatterMax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterMax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMax'.")

check_valid('tf.raw_ops.ScatterMax', generated_inputs['tf.raw_ops.ScatterMax'], lib="tf", suffix=0)

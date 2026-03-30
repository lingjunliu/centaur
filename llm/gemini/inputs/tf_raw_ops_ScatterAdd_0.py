
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_scatter_add_inputs():
    list_of_inputs = []

    # Input 1: Basic example with int32 ref, int32 indices, and int32 updates
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    indices = np.array([0, 2, 4], dtype=np.int32)
    updates = np.array([10, 20, 30], dtype=np.int32)
    use_locking = False
    name = None

    input_dict = {
        "ref": ref.value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 ref, int64 indices, and float32 updates
    ref = tf.Variable(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    indices = np.array([1, 3], dtype=np.int64)
    updates = np.array([10.0, 20.0], dtype=np.float32)
    use_locking = True
    name = "scatter_add_example_2"

    input_dict = {
        "ref": ref.value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64 ref, int32 indices, and int64 updates (duplicates)
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    indices = np.array([0, 0, 2], dtype=np.int32)
    updates = np.array([10, 20, 30], dtype=np.int64)
    use_locking = False
    name = "scatter_add_example_3"

    input_dict = {
        "ref": ref.value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D ref, 1D indices, and 2D updates
    ref = tf.Variable(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([[10, 20], [30, 40]], dtype=np.float32)
    use_locking = True
    name = "scatter_add_example_4"

    input_dict = {
        "ref": ref.value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 ref, int32 indices, and complex64 updates
    ref = tf.Variable(np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([10 + 10j, 30 + 30j], dtype=np.complex64)
    use_locking = False
    name = "scatter_add_example_5"

    input_dict = {
        "ref": ref.value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: qint8 ref, int32 indices, and qint8 updates
    ref = tf.Variable(np.array([1, 2, 3], dtype=np.int8))
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([10, 30], dtype=np.int8)
    use_locking = False
    name = "scatter_add_example_6"

    input_dict = {
        "ref": ref.value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7:  uint8 ref, int32 indices, and uint8 updates
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.uint8))
    indices = np.array([0, 2, 4], dtype=np.int32)
    updates = np.array([10, 20, 30], dtype=np.uint8)
    use_locking = False
    name = None

    input_dict = {
        "ref": ref.value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16 ref, int64 indices, and bfloat16 updates
    ref = tf.Variable(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16))
    indices = np.array([1, 3], dtype=np.int64)
    updates = np.array([10.0, 20.0], dtype=np.float16)
    use_locking = True
    name = "scatter_add_example_8"

    input_dict = {
        "ref": ref.value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint32 ref, int32 indices, and uint32 updates (duplicates)
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.uint32))
    indices = np.array([0, 0, 2], dtype=np.int32)
    updates = np.array([10, 20, 30], dtype=np.uint32)
    use_locking = False
    name = "scatter_add_example_9"

    input_dict = {
        "ref": ref.value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: half ref, int64 indices, and half updates
    ref = tf.Variable(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16))
    indices = np.array([1, 3], dtype=np.int64)
    updates = np.array([10.0, 20.0], dtype=np.float16)
    use_locking = False
    name = "scatter_add_example_10"

    input_dict = {
        "ref": ref.value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: uint64 ref, int32 indices, and uint64 updates (duplicates)
    ref = tf.Variable(np.array([1, 2, 3, 4, 5], dtype=np.uint64))
    indices = np.array([0, 0, 2], dtype=np.int32)
    updates = np.array([10, 20, 30], dtype=np.uint64)
    use_locking = False
    name = "scatter_add_example_11"

    input_dict = {
        "ref": ref.value(),
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterAdd"] = tf_raw_ops_scatter_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterAdd'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ScatterAdd', generated_inputs['tf.raw_ops.ScatterAdd'], lib="tf", suffix=0)

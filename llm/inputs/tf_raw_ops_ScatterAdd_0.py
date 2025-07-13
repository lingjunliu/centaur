
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_scatter_add_inputs():
    list_of_inputs = []

    # Input 1, valid
    ref = np.array([1, 2, 3, 4], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([5, 6], dtype=np.float32)
    use_locking = False
    name = "scatter_add_example_1"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    ref = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    updates = np.array([[7, 8], [9, 10]], dtype=np.int32)
    use_locking = True
    name = "scatter_add_example_2"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    indices = np.array([1], dtype=np.int64)
    updates = np.array([4.0], dtype=np.float64)
    use_locking = False
    name = "scatter_add_example_3"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, empty updates
    ref = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array([], dtype=np.int32)
    updates = np.array([], dtype=np.int32)
    use_locking = True
    name = "scatter_add_example_4"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, updates.shape = []
    ref = np.array([1, 2, 3], dtype=np.int32)
    indices = np.array([1], dtype=np.int32)
    updates = np.array(5, dtype=np.int32)
    use_locking = False
    name = "scatter_add_example_5"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, multi-dimensional indices
    ref = np.zeros((5, 5), dtype=np.float32)
    indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
    updates = np.array([1.0, 2.0], dtype=np.float32)
    use_locking = True
    name = "scatter_add_example_6"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7, valid, int64 ref
    ref = np.array([1, 2, 3, 4], dtype=np.int64)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([5, 6], dtype=np.int64)
    use_locking = False
    name = "scatter_add_example_7"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, uint8 ref
    ref = np.array([1, 2, 3, 4], dtype=np.uint8)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([5, 6], dtype=np.uint8)
    use_locking = False
    name = "scatter_add_example_8"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, half ref
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([5.0, 6.0], dtype=np.float16)
    use_locking = False
    name = "scatter_add_example_9"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, complex64 ref
    ref = np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex64)
    indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([5+5j, 6+6j], dtype=np.complex64)
    use_locking = False
    name = "scatter_add_example_10"

    input_dict = {
        "ref": ref,
        "indices": indices,
        "updates": updates,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ScatterAdd"] = tf_raw_ops_scatter_add_inputs()
for i in range(len(generated_inputs["tf.raw_ops.ScatterAdd"])):
  generated_inputs["tf.raw_ops.ScatterAdd"][i]["ref"] = tf.Variable(generated_inputs["tf.raw_ops.ScatterAdd"][i]["ref"])
  generated_inputs["tf.raw_ops.ScatterAdd"][i]["indices"] = tf.convert_to_tensor(generated_inputs["tf.raw_ops.ScatterAdd"][i]["indices"])
  generated_inputs["tf.raw_ops.ScatterAdd"][i]["updates"] = tf.convert_to_tensor(generated_inputs["tf.raw_ops.ScatterAdd"][i]["updates"])

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ScatterAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterAdd'.")

check_valid('tf.raw_ops.ScatterAdd', generated_inputs['tf.raw_ops.ScatterAdd'], lib="tf", suffix=0)

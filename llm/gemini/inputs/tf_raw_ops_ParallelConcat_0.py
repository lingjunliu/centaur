
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_parallel_concat_inputs():
    list_of_inputs = []

    # Input 1
    values = [np.array([[1]], dtype=np.int32), np.array([[2]], dtype=np.int32), np.array([[3]], dtype=np.int32)]
    shape = [3, 1]
    input_dict = {"values": [tf.convert_to_tensor(v, dtype=tf.int32) for v in values], "shape": shape, "name": "parallel_concat_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    values = [np.array([[1.0]], dtype=np.float32), np.array([[2.0]], dtype=np.float32), np.array([[3.0]], dtype=np.float32)]
    shape = [3, 1]
    input_dict = {"values": [tf.convert_to_tensor(v, dtype=tf.float32) for v in values], "shape": shape, "name": "parallel_concat_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    values = [np.array([[True]], dtype=np.bool_), np.array([[False]], dtype=np.bool_), np.array([[True]], dtype=np.bool_)]
    shape = [3, 1]
    input_dict = {"values": [tf.convert_to_tensor(v, dtype=tf.bool) for v in values], "shape": shape, "name": "parallel_concat_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    values = [np.array([[1, 2]], dtype=np.int32), np.array([[3, 4]], dtype=np.int32), np.array([[5, 6]], dtype=np.int32)]
    shape = [3, 2]
    input_dict = {"values": [tf.convert_to_tensor(v, dtype=tf.int32) for v in values], "shape": shape, "name": "parallel_concat_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    values = [np.array([[1.0, 2.0]], dtype=np.float64), np.array([[3.0, 4.0]], dtype=np.float64), np.array([[5.0, 6.0]], dtype=np.float64)]
    shape = [3, 2]
    input_dict = {"values": [tf.convert_to_tensor(v, dtype=tf.float64) for v in values], "shape": shape, "name": "parallel_concat_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    values = [np.array([[-1]], dtype=np.int32), np.array([[-2]], dtype=np.int32), np.array([[-3]], dtype=np.int32)]
    shape = [3, 1]
    input_dict = {"values": [tf.convert_to_tensor(v, dtype=tf.int32) for v in values], "shape": shape, "name": "parallel_concat_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    values = [np.array([[1]], dtype=np.int64), np.array([[2]], dtype=np.int64), np.array([[3]], dtype=np.int64)]
    shape = [3, 1]
    input_dict = {"values": [tf.convert_to_tensor(v, dtype=tf.int64) for v in values], "shape": shape, "name": "parallel_concat_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    values = [np.array([[1j]], dtype=np.complex64), np.array([[2j]], dtype=np.complex64), np.array([[3j]], dtype=np.complex64)]
    shape = [3, 1]
    input_dict = {"values": [tf.convert_to_tensor(v, dtype=tf.complex64) for v in values], "shape": shape, "name": "parallel_concat_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    values = [np.array([[1,2,3]], dtype=np.int32), np.array([[4,5,6]], dtype=np.int32), np.array([[7,8,9]], dtype=np.int32)]
    shape = [3, 3]
    input_dict = {"values": [tf.convert_to_tensor(v, dtype=tf.int32) for v in values], "shape": shape, "name": "parallel_concat_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    values = [np.array([[0.1]], dtype=np.float32), np.array([[0.2]], dtype=np.float32), np.array([[0.3]], dtype=np.float32)]
    shape = [3, 1]
    input_dict = {"values": [tf.convert_to_tensor(v, dtype=tf.float32) for v in values], "shape": shape, "name": "parallel_concat_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ParallelConcat"] = tf_raw_ops_parallel_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ParallelConcat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ParallelConcat'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ParallelConcat', generated_inputs['tf.raw_ops.ParallelConcat'], lib="tf", suffix=0)

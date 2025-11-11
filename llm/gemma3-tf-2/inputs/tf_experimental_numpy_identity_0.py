
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_identity_inputs():
    list_of_inputs = []
    
    n1 = np.int32(3)
    dtype1 = np.float32
    input_dict1 = {"n": n1, "dtype": dtype1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    n2 = np.int32(5)
    dtype2 = np.float64
    input_dict2 = {"n": n2, "dtype": dtype2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    n3 = np.int32(1)
    dtype3 = np.float16
    input_dict3 = {"n": n3, "dtype": dtype3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    n4 = np.int32(4)
    dtype4 = np.complex64
    input_dict4 = {"n": n4, "dtype": dtype4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    n5 = np.int32(2)
    dtype5 = np.complex128
    input_dict5 = {"n": n5, "dtype": dtype5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    n6 = np.int32(6)
    dtype6 = np.int32
    input_dict6 = {"n": n6, "dtype": dtype6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    n7 = np.int32(7)
    dtype7 = np.int64
    input_dict7 = {"n": n7, "dtype": dtype7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    n8 = np.int32(8)
    dtype8 = np.uint8
    input_dict8 = {"n": n8, "dtype": dtype8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    n9 = np.int32(9)
    dtype9 = np.uint16
    input_dict9 = {"n": n9, "dtype": dtype9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    n10 = np.int32(10)
    dtype10 = np.uint32
    input_dict10 = {"n": n10, "dtype": dtype10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.identity"] = tf_experimental_numpy_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.identity'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.identity', generated_inputs['tf.experimental.numpy.identity'], lib="tf", suffix=0)

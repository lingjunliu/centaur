
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_substr_inputs():
    list_of_inputs = []

    def create_tensor(data, dtype):
        return tf.constant(data, dtype=dtype)

    # Input 1
    input_tensor = np.array([b'Hello', b'World'])
    pos_tensor = np.array(1)
    len_tensor = np.array(3)
    unit_str = 'BYTE'
    name_str = None

    input_dict = {
        "input": create_tensor(input_tensor.astype(np.string_), tf.string),
        "pos": create_tensor(pos_tensor.astype(np.int32), tf.int32),
        "len": create_tensor(len_tensor.astype(np.int32), tf.int32),
        "unit": unit_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[b'ten', b'eleven', b'twelve'],
                            [b'thirteen', b'fourteen', b'fifteen'],
                            [b'sixteen', b'seventeen', b'eighteen']])
    pos_tensor = np.array([[1, 2, 3],
                           [1, 2, 3],
                           [1, 2, 3]])
    len_tensor = np.array([[2, 3, 4],
                           [4, 3, 2],
                           [5, 5, 5]])
    unit_str = 'BYTE'
    name_str = None

    input_dict = {
        "input": create_tensor(input_tensor.astype(np.string_), tf.string),
        "pos": create_tensor(pos_tensor.astype(np.int32), tf.int32),
        "len": create_tensor(len_tensor.astype(np.int32), tf.int32),
        "unit": unit_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[b'ten', b'eleven', b'twelve'],
                            [b'thirteen', b'fourteen', b'fifteen'],
                            [b'sixteen', b'seventeen', b'eighteen'],
                            [b'nineteen', b'twenty', b'twentyone']])
    pos_tensor = np.array([1, 2, 3])
    len_tensor = np.array([1, 2, 3])
    unit_str = 'BYTE'
    name_str = None

    input_dict = {
        "input": create_tensor(input_tensor.astype(np.string_), tf.string),
        "pos": create_tensor(pos_tensor.astype(np.int32), tf.int32),
        "len": create_tensor(len_tensor.astype(np.int32), tf.int32),
        "unit": unit_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array(b'thirteen')
    pos_tensor = np.array([1, 5, 7])
    len_tensor = np.array([3, 2, 1])
    unit_str = 'BYTE'
    name_str = None

    input_dict = {
        "input": create_tensor(input_tensor.astype(np.string_), tf.string),
        "pos": create_tensor(pos_tensor.astype(np.int32), tf.int32),
        "len": create_tensor(len_tensor.astype(np.int32), tf.int32),
        "unit": unit_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([b'Hello', b'World'])
    pos_tensor = np.array(-2)
    len_tensor = np.array(1)
    unit_str = 'BYTE'
    name_str = None

    input_dict = {
        "input": create_tensor(input_tensor.astype(np.string_), tf.string),
        "pos": create_tensor(pos_tensor.astype(np.int32), tf.int32),
        "len": create_tensor(len_tensor.astype(np.int32), tf.int32),
        "unit": unit_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([b'Hello', b'World'])
    pos_tensor = np.array(1)
    len_tensor = np.array(10)
    unit_str = 'BYTE'
    name_str = None

    input_dict = {
        "input": create_tensor(input_tensor.astype(np.string_), tf.string),
        "pos": create_tensor(pos_tensor.astype(np.int32), tf.int32),
        "len": create_tensor(len_tensor.astype(np.int32), tf.int32),
        "unit": unit_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([b'Hello', b'World'])
    pos_tensor = np.array(1)
    len_tensor = np.array(3)
    unit_str = 'UTF8_CHAR'
    name_str = None

    input_dict = {
        "input": create_tensor(input_tensor.astype(np.string_), tf.string),
        "pos": create_tensor(pos_tensor.astype(np.int32), tf.int32),
        "len": create_tensor(len_tensor.astype(np.int32), tf.int32),
        "unit": unit_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative pos
    input_tensor = np.array([b'Hello', b'World'])
    pos_tensor = np.array(-2)
    len_tensor = np.array(2)
    unit_str = 'BYTE'
    name_str = None

    input_dict = {
        "input": create_tensor(input_tensor.astype(np.string_), tf.string),
        "pos": create_tensor(pos_tensor.astype(np.int32), tf.int32),
        "len": create_tensor(len_tensor.astype(np.int32), tf.int32),
        "unit": unit_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([b'abcdef', b'ghijkl'])
    pos_tensor = np.array([1])
    len_tensor = np.array([2])
    unit_str = 'BYTE'
    name_str = None

    input_dict = {
        "input": create_tensor(input_tensor.astype(np.string_), tf.string),
        "pos": create_tensor(pos_tensor.astype(np.int32), tf.int32),
        "len": create_tensor(len_tensor.astype(np.int32), tf.int32),
        "unit": unit_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array(b'abcdef')
    pos_tensor = np.array([1, 2, 3])
    len_tensor = np.array([1, 1, 1])
    unit_str = 'BYTE'
    name_str = None

    input_dict = {
        "input": create_tensor(input_tensor.astype(np.string_), tf.string),
        "pos": create_tensor(pos_tensor.astype(np.int32), tf.int32),
        "len": create_tensor(len_tensor.astype(np.int32), tf.int32),
        "unit": unit_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11, int64 pos and len
    input_tensor = np.array([b'Hello', b'World'])
    pos_tensor = np.array(1)
    len_tensor = np.array(3)
    unit_str = 'BYTE'
    name_str = None

    input_dict = {
        "input": create_tensor(input_tensor.astype(np.string_), tf.string),
        "pos": create_tensor(pos_tensor.astype(np.int64), tf.int64),
        "len": create_tensor(len_tensor.astype(np.int64), tf.int64),
        "unit": unit_str,
        "name": name_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Substr"] = tf_raw_ops_substr_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Substr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Substr'.")

check_valid('tf.raw_ops.Substr', generated_inputs['tf.raw_ops.Substr'], lib="tf", suffix=0)

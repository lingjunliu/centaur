
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_OrderedMapClear_inputs():
    list_of_inputs = []

    input_dict = {
        "capacity": int(np.int64(0)),
        "memory_limit": int(np.int64(0)),
        "container": "",
        "shared_name": "",
        "name": "cleardefault",
        "dtypes": [np.dtype(np.int32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int32(10)),
        "memory_limit": int(np.int64(0)),
        "container": "mapa",
        "shared_name": "shareda",
        "name": "clearA",
        "dtypes": [np.dtype(np.float32), np.dtype(np.int64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(1000)),
        "memory_limit": int(np.int64(1024 * 1024)),
        "container": "containerascii",
        "shared_name": "sharedascii",
        "name": "opascii",
        "dtypes": [np.dtype(np.bool_), np.dtype(np.float64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(1)),
        "memory_limit": int(np.int64(1)),
        "container": "containeralpha",
        "shared_name": "sharedempty",
        "name": "n4",
        "dtypes": [np.dtype(np.complex64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(2**10)),
        "memory_limit": int(np.int64(0)),
        "container": "bigcapacity",
        "shared_name": "s5",
        "name": "n5",
        "dtypes": [np.dtype(np.uint8), np.dtype(np.int16)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(5)),
        "memory_limit": int(np.int64(100)),
        "container": "",
        "shared_name": "sharedonly",
        "name": "n6",
        "dtypes": [np.dtype(np.float16)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(0)),
        "memory_limit": int(np.int64(999999)),
        "container": "x" * 64,
        "shared_name": "y",
        "name": "zzz",
        "dtypes": [np.dtype(np.int8), np.dtype(np.float64), np.dtype(np.int64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int8(7)),
        "memory_limit": int(np.int16(0)),
        "container": "uintcontainer",
        "shared_name": "ushared",
        "name": "ucase",
        "dtypes": [np.dtype(np.uint16), np.dtype(np.uint32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(3)),
        "memory_limit": int(np.int64(3)),
        "container": "container3",
        "shared_name": "shared3",
        "name": "name3",
        "dtypes": [np.dtype(np.int64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(50)),
        "memory_limit": int(np.int64(0)),
        "container": "mixedtypes",
        "shared_name": "mix",
        "name": "mixclear",
        "dtypes": [np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.int32), np.dtype(np.int16)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(2)),
        "memory_limit": int(np.int64(2048)),
        "container": "complexc",
        "shared_name": "cx",
        "name": "cxclear",
        "dtypes": [np.dtype(np.complex128)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(8)),
        "memory_limit": int(np.int64(0)),
        "container": "boolcontainer",
        "shared_name": "bshared",
        "name": "boolclear",
        "dtypes": [np.dtype(np.bool_)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.OrderedMapClear"] = tf_raw_ops_OrderedMapClear_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.OrderedMapClear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.OrderedMapClear'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.OrderedMapClear', generated_inputs['tf.raw_ops.OrderedMapClear'], lib="tf", suffix=0)

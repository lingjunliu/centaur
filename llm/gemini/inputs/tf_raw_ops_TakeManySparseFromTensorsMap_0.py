
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_takemanysparsefromtensorsmap_inputs():
    list_of_inputs = []

    # The error "Unable to find SparseTensor" is a stateful, runtime error.
    # This operation requires a `SparseTensorsMap` to be populated by a preceding
    # operation like `AddSparseToTensorsMap`. In an isolated test environment,
    # the map is empty, so any attempt to look up a handle (which is required,
    # as N must be >= 1) will fail.
    # The previous errors have confirmed that `sparse_handles` must be a 1-D
    # tensor (vector) of shape `[N]`, where N >= 1.
    # The following inputs are syntactically valid according to the API's
    # signature and constraints, even though they are expected to fail with
    # the "Unable to find" error in an isolated test harness.

    # Input 1: Basic case with N=2 and dtype=float32
    input_dict = {
        'sparse_handles': np.array([101, 202], dtype=np.int64),
        'dtype': np.float32,
        'container': 'c1',
        'shared_name': 'map_A',
        'name': 'take_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different dtype (int32) and N=3
    input_dict = {
        'sparse_handles': np.array([10, 20, 30], dtype=np.int64),
        'dtype': np.int32,
        'container': 'c2',
        'shared_name': 'map_B',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger N (N=5), float64, and a container name
    input_dict = {
        'sparse_handles': np.array([1, 2, 3, 4, 5], dtype=np.int64),
        'dtype': np.float64,
        'container': 'my_container',
        'shared_name': 'shared_map_1',
        'name': 'take_large_n'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single handle (N=1)
    input_dict = {
        'sparse_handles': np.array([999], dtype=np.int64),
        'dtype': np.float32,
        'container': 'c4',
        'shared_name': 'single_handle_map',
        'name': 'take_one'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex dtype (complex64)
    input_dict = {
        'sparse_handles': np.array([77, 88], dtype=np.int64),
        'dtype': np.complex64,
        'container': 'complex_container',
        'shared_name': 'complex_map',
        'name': 'take_complex'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex dtype (complex128)
    input_dict = {
        'sparse_handles': np.array([111, 222, 333], dtype=np.int64),
        'dtype': np.complex128,
        'container': 'c6',
        'shared_name': 'complex_map_128',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large integer values for handles
    input_dict = {
        'sparse_handles': np.array([2**32, 2**33], dtype=np.int64),
        'dtype': np.float32,
        'container': 'c7',
        'shared_name': 'large_handle_map',
        'name': 'take_large_handles'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small integer dtype (int8)
    input_dict = {
        'sparse_handles': np.array([5, 15, 25], dtype=np.int64),
        'dtype': np.int8,
        'container': 'c8',
        'shared_name': 'int8_map',
        'name': 'take_int8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Unsigned integer dtype (uint8)
    input_dict = {
        'sparse_handles': np.array([1, 2], dtype=np.int64),
        'dtype': np.uint8,
        'container': 'another_container',
        'shared_name': 'another_shared_map',
        'name': 'take_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean dtype
    input_dict = {
        'sparse_handles': np.array([1000, 2000], dtype=np.int64),
        'dtype': np.bool_,
        'container': 'c10',
        'shared_name': 'bool_map',
        'name': 'take_bools'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.TakeManySparseFromTensorsMap"] = get_tf_raw_ops_takemanysparsefromtensorsmap_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.TakeManySparseFromTensorsMap' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TakeManySparseFromTensorsMap'.")

check_valid('tf.raw_ops.TakeManySparseFromTensorsMap', generated_inputs['tf.raw_ops.TakeManySparseFromTensorsMap'], lib="tf", suffix=0)

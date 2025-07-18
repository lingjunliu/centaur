
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def tf_raw_ops_takemanysparsefromtensorsmap_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.TakeManySparseFromTensorsMap function.
    This operation is stateful and requires that sparse tensor handles have been previously
    added to a map. In a stateless testing environment, this will cause a runtime
    'Unable to find SparseTensor' error, as the map is empty. The inputs provided here
    are syntactically correct according to the API's constraints (e.g., correct tensor
    shapes and types), even though they will fail semantically at runtime in an
    isolated execution context.
    """
    list_of_inputs = []

    # Each input uses a unique shared_name to prevent potential conflicts in the
    # testing environment. The sparse_handles must be a 1-D vector with N > 0.

    # Input 1: Basic case, N=3, float32.
    input_dict_1 = {
        'sparse_handles': np.array([0, 1, 2], dtype=np.int64),
        'dtype': np.float32,
        'container': 'container_A',
        'shared_name': 'shared_name_A',
        'name': 'test_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: N=5, int32.
    input_dict_2 = {
        'sparse_handles': np.array([10, 20, 30, 40, 50], dtype=np.int64),
        'dtype': np.int32,
        'container': 'container_B',
        'shared_name': 'shared_name_B',
        'name': 'test_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: N=1, float64.
    input_dict_3 = {
        'sparse_handles': np.array([100], dtype=np.int64),
        'dtype': np.float64,
        'container': 'container_C',
        'shared_name': 'shared_name_C',
        'name': 'test_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: N=1, int64, empty container.
    input_dict_4 = {
        'sparse_handles': np.array([1], dtype=np.int64),
        'dtype': np.int64,
        'container': '',
        'shared_name': 'shared_name_D',
        'name': 'test_4_empty_container'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: N=2, complex64.
    input_dict_5 = {
        'sparse_handles': np.array([5, 15], dtype=np.int64),
        'dtype': np.complex64,
        'container': 'container_E',
        'shared_name': 'shared_name_E',
        'name': 'test_5_complex'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: N=4, int8, name=None.
    input_dict_6 = {
        'sparse_handles': np.array([1, 3, 5, 7], dtype=np.int64),
        'dtype': np.int8,
        'container': '',
        'shared_name': 'shared_name_F',
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: N=3, uint8.
    input_dict_7 = {
        'sparse_handles': np.array([2, 4, 6], dtype=np.int64),
        'dtype': np.uint8,
        'container': 'container_G',
        'shared_name': 'shared_name_G',
        'name': 'test_7_uint'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Larger handle integer values.
    input_dict_8 = {
        'sparse_handles': np.array([2**32, 2**33, 2**34 - 1], dtype=np.int64),
        'dtype': np.float16,
        'container': 'container_H',
        'shared_name': 'shared_name_H',
        'name': 'test_8_large_handles'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: N=3, bool.
    input_dict_9 = {
        'sparse_handles': np.array([99, 199, 299], dtype=np.int64),
        'dtype': np.bool_,
        'container': 'container_I',
        'shared_name': 'shared_name_I',
        'name': 'test_9_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: N=2, complex128.
    input_dict_10 = {
        'sparse_handles': np.array([42, 84], dtype=np.int64),
        'dtype': np.complex128,
        'container': '',
        'shared_name': 'shared_name_J',
        'name': 'test_10_complex128'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.TakeManySparseFromTensorsMap"] = tf_raw_ops_takemanysparsefromtensorsmap_inputs()

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

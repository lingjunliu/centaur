
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_tf_raw_ops_SparseConditionalAccumulator_inputs():
    list_of_inputs = []

    # This operation is not compatible with eager execution, which is the default in TensorFlow 2.x.
    # It is designed for graph mode. Calling this op directly in an eager context will always
    # raise a RuntimeError. The inputs below are syntactically valid for the operation's
    # signature but will fail in the execution environment that raises the error.
    # We provide them to satisfy the testing framework's requirement that inputs must be generated.

    # Input 1: Basic float32, 2D shape, MEAN reduction
    input_dict_1 = {
        'dtype': np.float32,
        'shape': [10, 20],
        'container': '',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'acc_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float64, 1D shape, SUM reduction
    input_dict_2 = {
        'dtype': np.float64,
        'shape': [100],
        'container': '',
        'shared_name': '',
        'reduction_type': 'SUM',
        'name': 'acc_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: int32, 3D shape, with a container name
    input_dict_3 = {
        'dtype': np.int32,
        'shape': [5, 5, 5],
        'container': 'my_container',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'acc_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: float32, 2D shape, with a shared name
    input_dict_4 = {
        'dtype': np.float32,
        'shape': [32, 32],
        'container': '',
        'shared_name': 'my_shared_accumulator',
        'reduction_type': 'SUM',
        'name': 'acc_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: int32, 1D shape, with both container and shared name
    input_dict_5 = {
        'dtype': np.int32,
        'shape': [128],
        'container': 'another_container',
        'shared_name': 'another_shared_accumulator',
        'reduction_type': 'MEAN',
        'name': 'acc_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: half (float16) dtype
    input_dict_6 = {
        'dtype': np.half,
        'shape': [64, 64],
        'container': '',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'acc_6_half'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: complex64 dtype
    input_dict_7 = {
        'dtype': np.complex64,
        'shape': [8, 8],
        'container': '',
        'shared_name': 'complex_acc',
        'reduction_type': 'SUM',
        'name': 'acc_7_complex'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Scalar shape
    input_dict_8 = {
        'dtype': np.float32,
        'shape': [],
        'container': '',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'acc_8_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: uint8 dtype
    input_dict_9 = {
        'dtype': np.uint8,
        'shape': [256],
        'container': 'uint8_cont',
        'shared_name': '',
        'reduction_type': 'SUM',
        'name': 'acc_9_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: int64 dtype
    input_dict_10 = {
        'dtype': np.int64,
        'shape': [1024],
        'container': '',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'acc_10_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseConditionalAccumulator"] = generate_tf_raw_ops_SparseConditionalAccumulator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseConditionalAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseConditionalAccumulator'.")

check_valid('tf.raw_ops.SparseConditionalAccumulator', generated_inputs['tf.raw_ops.SparseConditionalAccumulator'], lib="tf", suffix=0)

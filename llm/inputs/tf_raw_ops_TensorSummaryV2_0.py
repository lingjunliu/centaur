
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
from tensorflow.core.framework import summary_pb2

def tf_raw_ops_TensorSummaryV2_inputs():
    list_of_inputs = []

    # Create a serialized empty SummaryMetadata proto.
    # This avoids the problematic protobuf field manipulation that seems to fail in the execution environment.
    empty_metadata = summary_pb2.SummaryMetadata()
    serialized_empty_metadata = np.array(empty_metadata.SerializeToString())

    # Case 1: Scalar float tensor
    input_dict_1 = {
        'name': 'ScalarFloatSummary',
        'tag': np.array('scalar_float'),
        'tensor': np.array(3.14, dtype=np.float32),
        'serialized_summary_metadata': serialized_empty_metadata
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: 1D integer tensor with negative values
    input_dict_2 = {
        'name': None,
        'tag': np.array('vector_int_summary'),
        'tensor': np.array([-1, 0, 1, -2, 2], dtype=np.int32),
        'serialized_summary_metadata': serialized_empty_metadata
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: 2D float64 tensor
    input_dict_3 = {
        'name': 'MatrixFloat64Summary',
        'tag': np.array('matrix_summary/float64'),
        'tensor': np.array([[-10.5, 20.0], [0.0, -0.5]], dtype=np.float64),
        'serialized_summary_metadata': serialized_empty_metadata
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: 3D uint8 tensor
    input_dict_4 = {
        'name': 'ImageSummary',
        'tag': np.array('image_summary/rgb'),
        'tensor': np.arange(24, dtype=np.uint8).reshape((2, 4, 3)),
        'serialized_summary_metadata': serialized_empty_metadata
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: 0-D (scalar) int64 tensor
    input_dict_5 = {
        'name': None,
        'tag': np.array('scalar_int64'),
        'tensor': np.array(123456789012345, dtype=np.int64),
        'serialized_summary_metadata': serialized_empty_metadata
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: 1D complex64 tensor
    input_dict_6 = {
        'name': 'ComplexSummary',
        'tag': np.array('complex_vector'),
        'tensor': np.array([1+2j, 3-4j, -5+6j], dtype=np.complex64),
        'serialized_summary_metadata': serialized_empty_metadata
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Empty 1D tensor
    input_dict_7 = {
        'name': None,
        'tag': np.array('empty_tensor_summary'),
        'tensor': np.array([], dtype=np.float32),
        'serialized_summary_metadata': serialized_empty_metadata
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: High-rank tensor (4D) with float16
    input_dict_8 = {
        'name': '4DTensorSummary',
        'tag': np.array('4d_tensor'),
        'tensor': np.random.rand(2, 2, 2, 2).astype(np.float16),
        'serialized_summary_metadata': serialized_empty_metadata
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: 2D boolean tensor
    input_dict_9 = {
        'name': 'BooleanMatrix',
        'tag': np.array('bool_matrix'),
        'tensor': np.array([[True, False], [False, True]], dtype=np.bool_),
        'serialized_summary_metadata': serialized_empty_metadata
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Tensor with special float values (NaN, Inf)
    input_dict_10 = {
        'name': None,
        'tag': np.array('special_floats'),
        'tensor': np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32),
        'serialized_summary_metadata': serialized_empty_metadata
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Case 11: String tensor (bytes) for the main tensor argument
    input_dict_11 = {
        'name': 'StringTensorSummary',
        'tag': np.array('string_tensor'),
        'tensor': np.array([b'hello', b'world']),
        'serialized_summary_metadata': serialized_empty_metadata
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Case 12: Large 1D tensor
    input_dict_12 = {
        'name': 'LargeVector',
        'tag': np.array('large_vector'),
        'tensor': np.arange(1000, dtype=np.int16),
        'serialized_summary_metadata': serialized_empty_metadata
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.raw_ops.TensorSummaryV2"] = tf_raw_ops_TensorSummaryV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.TensorSummaryV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TensorSummaryV2'.")

check_valid('tf.raw_ops.TensorSummaryV2', generated_inputs['tf.raw_ops.TensorSummaryV2'], lib="tf", suffix=0)

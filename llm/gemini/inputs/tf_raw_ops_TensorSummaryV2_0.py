
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
from tensorflow.core.framework import summary_pb2

def tf_raw_ops_TensorSummaryV2_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.TensorSummaryV2 function.
    """
    list_of_inputs = []

    # Helper function to create a serialized empty SummaryMetadata proto.
    # The dtype is set to object to avoid issues with fixed-length string dtypes ('S...').
    def create_empty_metadata():
        metadata = summary_pb2.SummaryMetadata()
        # The dtype=object ensures compatibility with type checkers that may not
        # recognize specific fixed-length string dtypes ('S...').
        return np.array(metadata.SerializeToString(), dtype=object)

    # Input 1: Basic scalar int tensor
    input_dict = {
        'tag': np.array(b'scalar_int_summary', dtype=object),
        'tensor': np.array(42, dtype=np.int32),
        'serialized_summary_metadata': create_empty_metadata(),
        'name': 'ScalarIntTest'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float tensor with negative values
    input_dict = {
        'tag': np.array(b'1d_float_summary', dtype=object),
        'tensor': np.array([-1.1, 0.0, 2.2, -3.3], dtype=np.float32),
        'serialized_summary_metadata': create_empty_metadata(),
        'name': '1DFloatTest'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D boolean tensor
    input_dict = {
        'tag': np.array(b'2d_bool_summary', dtype=object),
        'tensor': np.array([[True, False], [False, True]], dtype=np.bool_),
        'serialized_summary_metadata': create_empty_metadata(),
        'name': '2DBoolTest'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int64 tensor
    input_dict = {
        'tag': np.array(b'3d_int64_summary', dtype=object),
        'tensor': np.arange(24, dtype=np.int64).reshape((2, 3, 4)),
        'serialized_summary_metadata': create_empty_metadata(),
        'name': '3DInt64Test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty tensor
    input_dict = {
        'tag': np.array(b'empty_tensor_summary', dtype=object),
        'tensor': np.array([], dtype=np.float32),
        'serialized_summary_metadata': create_empty_metadata(),
        'name': 'EmptyTensorTest'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float16 tensor with name=None
    input_dict = {
        'tag': np.array(b'float16_summary', dtype=object),
        'tensor': np.array([1.0, 2.5, 3.0], dtype=np.float16),
        'serialized_summary_metadata': create_empty_metadata(),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large 2D tensor (uint8, like an image)
    input_dict = {
        'tag': np.array(b'image_summary', dtype=object),
        'tensor': np.arange(100, dtype=np.uint8).reshape((10, 10)),
        'serialized_summary_metadata': create_empty_metadata(),
        'name': 'ImageSummaryTest'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex number tensor
    input_dict = {
        'tag': np.array(b'complex_summary', dtype=object),
        'tensor': np.array([1+2j, 3-4j, 5+0j], dtype=np.complex64),
        'serialized_summary_metadata': create_empty_metadata(),
        'name': 'ComplexTest'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Scalar string tensor (for the 'tensor' argument)
    input_dict = {
        'tag': np.array(b'text_summary', dtype=object),
        'tensor': np.array(b'This is a test string.', dtype=object),
        'serialized_summary_metadata': create_empty_metadata(),
        'name': 'TextSummaryTest'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with all zeros
    input_dict = {
        'tag': np.array(b'zeros_summary', dtype=object),
        'tensor': np.zeros((3, 3), dtype=np.int8),
        'serialized_summary_metadata': create_empty_metadata(),
        'name': 'ZerosTest'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D tensor with double precision floats
    input_dict = {
        'tag': np.array(b'4d_double_summary', dtype=object),
        'tensor': np.random.rand(1, 2, 2, 1).astype(np.float64),
        'serialized_summary_metadata': create_empty_metadata(),
        'name': '4dDoubleTest'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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

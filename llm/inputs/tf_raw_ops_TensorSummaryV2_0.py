
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_TensorSummaryV2_inputs():
    list_of_inputs = []

    # Input 1
    tag = np.array("example_tag_1", dtype=np.bytes_)
    tensor = np.array([1, 2, 3], dtype=np.int32)
    serialized_summary_metadata = np.array("metadata_1", dtype=np.bytes_)
    input_dict = {"tag": tag, "tensor": tensor, "serialized_summary_metadata": serialized_summary_metadata, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tag = np.array("example_tag_2", dtype=np.bytes_)
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    serialized_summary_metadata = np.array("metadata_2", dtype=np.bytes_)
    input_dict = {"tag": tag, "tensor": tensor, "serialized_summary_metadata": serialized_summary_metadata, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tag = np.array("example_tag_3", dtype=np.bytes_)
    tensor = np.array([True, False, True], dtype=np.bool_)
    serialized_summary_metadata = np.array("metadata_3", dtype=np.bytes_)
    input_dict = {"tag": tag, "tensor": tensor, "serialized_summary_metadata": serialized_summary_metadata, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tag = np.array("example_tag_4", dtype=np.bytes_)
    tensor = np.array([b"hello", b"world"], dtype=np.object_)
    serialized_summary_metadata = np.array("metadata_4", dtype=np.bytes_)
    input_dict = {"tag": tag, "tensor": tensor, "serialized_summary_metadata": serialized_summary_metadata, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tag = np.array("example_tag_5", dtype=np.bytes_)
    tensor = np.array([1, -2, 3, -4], dtype=np.int64)
    serialized_summary_metadata = np.array("metadata_5", dtype=np.bytes_)
    input_dict = {"tag": tag, "tensor": tensor, "serialized_summary_metadata": serialized_summary_metadata, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tag = np.array("example_tag_6", dtype=np.bytes_)
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    serialized_summary_metadata = np.array("metadata_6", dtype=np.bytes_)
    input_dict = {"tag": tag, "tensor": tensor, "serialized_summary_metadata": serialized_summary_metadata, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tag = np.array("example_tag_7", dtype=np.bytes_)
    tensor = np.array(1.23456789, dtype=np.float64)
    serialized_summary_metadata = np.array("metadata_7", dtype=np.bytes_)
    input_dict = {"tag": tag, "tensor": tensor, "serialized_summary_metadata": serialized_summary_metadata, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tag = np.array("example_tag_8", dtype=np.bytes_)
    tensor = np.array([1+1j, 2+2j], dtype=np.complex64)
    serialized_summary_metadata = np.array("metadata_8", dtype=np.bytes_)
    input_dict = {"tag": tag, "tensor": tensor, "serialized_summary_metadata": serialized_summary_metadata, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tag = np.array("example_tag_9", dtype=np.bytes_)
    tensor = np.array([], dtype=np.int32)
    serialized_summary_metadata = np.array("metadata_9", dtype=np.bytes_)
    input_dict = {"tag": tag, "tensor": tensor, "serialized_summary_metadata": serialized_summary_metadata, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tag = np.array("example_tag_10", dtype=np.bytes_)
    tensor = np.array([1, 2, 3], dtype=np.uint8)
    serialized_summary_metadata = np.array("metadata_10", dtype=np.bytes_)
    input_dict = {"tag": tag, "tensor": tensor, "serialized_summary_metadata": serialized_summary_metadata, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    tag = np.array(b"example_tag_11")
    tensor = np.zeros((2, 3, 4), dtype=np.float32)
    serialized_summary_metadata = np.array(b"metadata_11")
    input_dict = {"tag": tag, "tensor": tensor, "serialized_summary_metadata": serialized_summary_metadata, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.TensorSummaryV2"] = tf_raw_ops_TensorSummaryV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.TensorSummaryV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TensorSummaryV2'.")

check_valid('tf.raw_ops.TensorSummaryV2', generated_inputs['tf.raw_ops.TensorSummaryV2'], lib="tf", suffix=0)

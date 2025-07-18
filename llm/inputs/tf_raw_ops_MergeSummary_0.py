
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
from tensorflow.core.framework import summary_pb2

def _create_scalar_summary(tag, value):
    """Helper to create a serialized scalar summary proto."""
    summary = summary_pb2.Summary(
        value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(value))]
    )
    return summary.SerializeToString()

def _create_image_summary(tag):
    """Helper to create a serialized image summary proto with dummy data."""
    image_data = summary_pb2.Summary.Image(
        height=1, width=1, colorspace=3, encoded_image_string=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!@\x14\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    summary = summary_pb2.Summary(
        value=[summary_pb2.Summary.Value(tag=tag, image=image_data)]
    )
    return summary.SerializeToString()

def _create_empty_summary():
    """Helper to create a serialized empty summary proto."""
    return summary_pb2.Summary().SerializeToString()

def get_tf_raw_ops_mergesummary_inputs():
    list_of_inputs = []

    # Case 1: Single input tensor with a single summary
    s1 = _create_scalar_summary("case1/scalar", 100.0)
    list_of_inputs.append(
        {'inputs': [np.array([s1], dtype=np.object_)],
         'name': 'single_tensor_single_summary'}
    )

    # Case 2: Single input tensor with multiple summaries
    s2a = _create_scalar_summary("case2/scalar_a", -5.5)
    s2b = _create_scalar_summary("case2/scalar_b", 20.2)
    list_of_inputs.append(
        {'inputs': [np.array([s2a, s2b], dtype=np.object_)],
         'name': 'single_tensor_multiple_summaries'}
    )

    # Case 3: Multiple input tensors, each with one summary
    s3a = _create_scalar_summary("case3/metric1", 3.14)
    s3b = _create_scalar_summary("case3/metric2", 2.718)
    list_of_inputs.append(
        {'inputs': [np.array([s3a], dtype=np.object_), np.array([s3b], dtype=np.object_)],
         'name': 'multiple_tensors'}
    )

    # Case 4: Scalar input tensor (shape ())
    s4 = _create_scalar_summary("case4/scalar", 4.0)
    list_of_inputs.append(
        {'inputs': [np.array(s4, dtype=np.object_)],
         'name': 'scalar_tensor_input'}
    )

    # Case 5: 2D input tensor
    s5a = _create_scalar_summary("case5/s_00", 1.0)
    s5b = _create_scalar_summary("case5/s_01", 2.0)
    s5c = _create_scalar_summary("case5/s_10", 3.0)
    s5d = _create_scalar_summary("case5/s_11", 4.0)
    list_of_inputs.append(
        {'inputs': [np.array([[s5a, s5b], [s5c, s5d]], dtype=np.object_)],
         'name': '2d_tensor_input'}
    )

    # Case 6: Mixed shapes in input tensors list
    s6a = _create_scalar_summary("case6/scalar_tensor", 6.0)
    s6b = _create_scalar_summary("case6/vec_tensor_1", 6.1)
    s6c = _create_scalar_summary("case6/vec_tensor_2", 6.2)
    list_of_inputs.append(
        {'inputs': [np.array(s6a, dtype=np.object_), np.array([s6b, s6c], dtype=np.object_)],
         'name': 'mixed_shape_inputs'}
    )

    # Case 7: Input containing an empty summary proto
    s7a = _create_scalar_summary("case7/valid_summary", 7.0)
    s7b = _create_empty_summary()
    list_of_inputs.append(
        {'inputs': [np.array([s7a, s7b], dtype=np.object_)],
         'name': 'with_empty_summary'}
    )

    # Case 8: Mixed summary types (scalar and image)
    s8a = _create_scalar_summary("case8/loss", 0.123)
    s8b = _create_image_summary("case8/input_image")
    list_of_inputs.append(
        {'inputs': [np.array([s8a], dtype=np.object_), np.array([s8b], dtype=np.object_)],
         'name': 'mixed_summary_types'}
    )

    # Case 9: Operation name is None (optional)
    s9 = _create_scalar_summary("case9/value", 9.9)
    list_of_inputs.append(
        {'inputs': [np.array([s9], dtype=np.object_)],
         'name': None}
    )

    # Case 10: 3D input tensor
    summaries_10 = [_create_scalar_summary(f"case10/s_{i}", 10.0 + i) for i in range(8)]
    input_tensor_10 = np.array(summaries_10, dtype=np.object_).reshape((2, 2, 2))
    list_of_inputs.append(
        {'inputs': [input_tensor_10],
         'name': '3d_tensor_input'}
    )
    
    # Case 11: A larger number of input tensors in the list
    summaries_11 = [_create_scalar_summary(f"case11/s_{i}", 11.0 + i) for i in range(5)]
    input_tensors_11 = [np.array([s], dtype=np.object_) for s in summaries_11]
    list_of_inputs.append(
        {'inputs': input_tensors_11,
         'name': 'many_input_tensors'}
    )

    # Case 12: Unicode tag in summary
    s12 = _create_scalar_summary("case12/résumé_scalaire", 12.0)
    list_of_inputs.append(
        {'inputs': [np.array([s12], dtype=np.object_)],
         'name': 'unicode_tag'}
    )
    
    return list_of_inputs

generated_inputs["tf.raw_ops.MergeSummary"] = get_tf_raw_ops_mergesummary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MergeSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MergeSummary'.")

check_valid('tf.raw_ops.MergeSummary', generated_inputs['tf.raw_ops.MergeSummary'], lib="tf", suffix=0)

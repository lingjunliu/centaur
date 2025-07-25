
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
from tensorflow.core.framework import summary_pb2

def _create_serialized_summary(tags_and_values):
  """Helper to create a serialized Summary proto from (tag, value) pairs."""
  summary = summary_pb2.Summary()
  for tag, value in tags_and_values:
    summary.value.add(tag=tag, simple_value=float(value))
  return summary.SerializeToString()

def generate_tf_raw_ops_mergesummary_inputs():
  """
  Generates a list of valid inputs for tf.raw_ops.MergeSummary.
  Based on the error analysis, the 'tensor_list' type for the 'inputs'
  parameter is expected by the testing framework to be a single 1D numpy array,
  where each element is a string representing a serialized summary.
  The dtype is set to 'object' to handle variable-length strings correctly.
  """
  list_of_inputs = []

  # Case 1: A single summary.
  s1 = _create_serialized_summary([("tag1", 1.0)])
  list_of_inputs.append({
      "inputs": np.array([s1], dtype=object),
      "name": "single_summary"
  })

  # Case 2: Two distinct summaries.
  s2 = _create_serialized_summary([("tag2", 2.0)])
  s3 = _create_serialized_summary([("tag3", 3.0)])
  list_of_inputs.append({
      "inputs": np.array([s2, s3], dtype=object),
      "name": "two_summaries"
  })

  # Case 3: A summary proto that contains multiple values.
  s4_multi = _create_serialized_summary([("tag4.1", 4.1), ("tag4.2", 4.2)])
  s5 = _create_serialized_summary([("tag5", 5.0)])
  list_of_inputs.append({
      "inputs": np.array([s4_multi, s5], dtype=object),
      "name": "multi_value_in_proto"
  })

  # Case 4: A larger number of summaries (10).
  s_list_10 = [
      _create_serialized_summary([(f"tag_10_{i}", i * 1.1)]) for i in range(10)
  ]
  list_of_inputs.append({
      "inputs": np.array(s_list_10, dtype=object),
      "name": "ten_summaries"
  })

  # Case 5: Summaries with negative and zero values.
  s_neg = _create_serialized_summary([("neg_tag", -99.9)])
  s_zero = _create_serialized_summary([("zero_tag", 0.0)])
  s_pos = _create_serialized_summary([("pos_tag", 99.9)])
  list_of_inputs.append({
      "inputs": np.array([s_neg, s_zero, s_pos], dtype=object),
      "name": "varied_sign_values"
  })

  # Case 6: One of the summaries is an empty proto.
  s_empty = summary_pb2.Summary().SerializeToString()
  s7 = _create_serialized_summary([("tag7", 7.0)])
  list_of_inputs.append({
      "inputs": np.array([s_empty, s7], dtype=object),
      "name": "with_empty_summary_proto"
  })

  # Case 7: No optional name provided (name=None).
  s8 = _create_serialized_summary([("tag8", 8.0)])
  list_of_inputs.append({
      "inputs": np.array([s8], dtype=object),
      "name": None
  })

  # Case 8: Long tag name to ensure variable string sizes are handled.
  long_tag = "a_very_long_tag_name_to_test_string_serialization_and_parsing_correctly"
  s9 = _create_serialized_summary([(long_tag, 9.0)])
  list_of_inputs.append({
      "inputs": np.array([s9], dtype=object),
      "name": "long_tag_name"
  })

  # Case 9: Input tensor with shape (1,).
  s10 = _create_serialized_summary([("tag10", 10.0)])
  list_of_inputs.append({
      "inputs": np.array([s10], dtype=object).reshape(1,),
      "name": "explicit_1d_shape"
  })

  # Case 10: Input tensor with shape (5,).
  s_list_5 = [
      _create_serialized_summary([(f"tag_5_{i}", float(i))]) for i in range(5)
  ]
  list_of_inputs.append({
      "inputs": np.array(s_list_5, dtype=object),
      "name": "five_summaries"
  })

  # Case 11: A single tensor of summaries, but reshaped from 2D. The final tensor is 1D.
  # This tests that the origin of the data doesn't matter, only its final shape.
  s_2d_list = [
      _create_serialized_summary([(f"2d_source_tag_{r}_{c}", r*2+c)]) for r in range(2) for c in range(2)
  ]
  list_of_inputs.append({
      "inputs": np.array(s_2d_list, dtype=object).flatten(),
      "name": "flattened_2d_summaries"
  })
  
  # Case 12: A large number of summaries (50).
  s_list_50 = [
      _create_serialized_summary([(f"tag_50_{i}", i * 0.1)]) for i in range(50)
  ]
  list_of_inputs.append({
      "inputs": np.array(s_list_50, dtype=object),
      "name": "fifty_summaries"
  })

  return [copy.deepcopy(d) for d in list_of_inputs]

generated_inputs["tf.raw_ops.MergeSummary"] = generate_tf_raw_ops_mergesummary_inputs()

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

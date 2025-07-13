
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MergeSummary_inputs():
    list_of_inputs = []

    # Input 1: Empty summary
    inputs = [tf.constant(b"")]
    input_dict = {"inputs": inputs, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single simple summary
    summary_str = tf.compat.as_bytes("""
    value {
      tag: "simple_value"
      simple_value: 1.0
    }
    """)
    inputs = [tf.constant(summary_str)]
    input_dict = {"inputs": inputs, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple simple summaries
    summary_str1 = tf.compat.as_bytes("""
    value {
      tag: "simple_value_1"
      simple_value: 1.0
    }
    """)
    summary_str2 = tf.compat.as_bytes("""
    value {
      tag: "simple_value_2"
      simple_value: 2.0
    }
    """)
    inputs = [tf.constant(summary_str1), tf.constant(summary_str2)]
    input_dict = {"inputs": inputs, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Summaries with different types of values
    summary_str1 = tf.compat.as_bytes("""
    value {
      tag: "simple_value"
      simple_value: 1.0
    }
    """)
    summary_str2 = tf.compat.as_bytes("""
    value {
      tag: "tensor_value"
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        float_val: [1.0, 2.0, 3.0, 4.0]
      }
    }
    """)
    inputs = [tf.constant(summary_str1), tf.constant(summary_str2)]
    input_dict = {"inputs": inputs, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Summaries with histograms
    summary_str1 = tf.compat.as_bytes("""
    value {
      tag: "histogram_value"
      histo {
        min: 0.0
        max: 1.0
        num: 1.0
        sum: 0.5
        sum_squares: 0.25
        bucket_limit: [1.0]
        bucket: [1.0]
      }
    }
    """)
    inputs = [tf.constant(summary_str1)]
    input_dict = {"inputs": inputs, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Summaries with images
    summary_str1 = tf.compat.as_bytes("""
    value {
      tag: "image_value"
      image {
        height: 1
        width: 1
        colorspace: 1
        encoded_image_string: "abc"
      }
    }
    """)
    inputs = [tf.constant(summary_str1)]
    input_dict = {"inputs": inputs, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 7: Summary with audio
    summary_str1 = tf.compat.as_bytes("""
    value {
      tag: "audio_value"
      audio {
        sample_rate: 44100.0
        num_channels: 1
        length_secs: 1.0
        encoded_audio_string: "abc"
        content_type: "audio/wav"
      }
    }
    """)
    inputs = [tf.constant(summary_str1)]
    input_dict = {"inputs": inputs, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Summary with metadata
    summary_str1 = tf.compat.as_bytes("""
    value {
      tag: "scalar_value"
      metadata {
        plugin_data {
          plugin_name: "scalars"
          content: "abc"
        }
      }
      simple_value: 1.0
    }
    """)
    inputs = [tf.constant(summary_str1)]
    input_dict = {"inputs": inputs, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: multiple inputs, with different names
    summary_str1 = tf.compat.as_bytes("""
    value {
      tag: "scalar_value1"
      simple_value: 1.0
    }
    """)
    summary_str2 = tf.compat.as_bytes("""
    value {
      tag: "scalar_value2"
      simple_value: 2.0
    }
    """)
    inputs = [tf.constant(summary_str1), tf.constant(summary_str2)]
    input_dict = {"inputs": inputs, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: with name
    summary_str1 = tf.compat.as_bytes("""
    value {
      tag: "scalar_value1"
      simple_value: 1.0
    }
    """)
    inputs = [tf.constant(summary_str1)]
    input_dict = {"inputs": inputs, "name": "my_summary"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MergeSummary"] = tf_raw_ops_MergeSummary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MergeSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MergeSummary'.")

check_valid('tf.raw_ops.MergeSummary', generated_inputs['tf.raw_ops.MergeSummary'], lib="tf", suffix=0)

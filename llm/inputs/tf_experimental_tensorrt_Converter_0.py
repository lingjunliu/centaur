
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_tensorrt_converter_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'input_saved_model_dir': 'path/to/model_1',
        'input_saved_model_tags': ['serve'],
        'input_saved_model_signature_key': 'serving_default',
        'use_dynamic_shape': False,
        'dynamic_shape_profile_strategy': '',
        'max_workspace_size_bytes': 2147483648,
        'precision_mode': 'FP32',
        'minimum_segment_size': 5,
        'maximum_cached_engines': 2,
        'use_calibration': False,
        'allow_build_at_runtime': True,
        'conversion_params': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'input_saved_model_dir': 'path/to/model_2',
        'input_saved_model_tags': ['gpu'],
        'input_saved_model_signature_key': 'my_signature',
        'use_dynamic_shape': True,
        'dynamic_shape_profile_strategy': 'Range',
        'max_workspace_size_bytes': 536870912,
        'precision_mode': 'FP16',
        'minimum_segment_size': 2,
        'maximum_cached_engines': 4,
        'use_calibration': True,
        'allow_build_at_runtime': False,
        'conversion_params': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs["tf.experimental.tensorrt.Converter"] = tf_experimental_tensorrt_converter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.tensorrt.Converter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.tensorrt.Converter'.")

check_valid('tf.experimental.tensorrt.Converter', generated_inputs['tf.experimental.tensorrt.Converter'], lib="tf", suffix=0)

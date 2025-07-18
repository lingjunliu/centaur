
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_tensorrt_converter_inputs():
    # This function generates inputs for tf.experimental.tensorrt.Converter.
    # The API requires a TensorFlow build with TensorRT support. If the
    # execution environment lacks this, a RuntimeError will be raised upon
    # instantiating the Converter, regardless of the parameters provided.
    # The following inputs are valid according to the API's signature but
    # will still fail in an environment without TensorRT support.
    list_of_inputs = []

    # Input 1: Basic FP32 conversion.
    input_dict_1 = {
        'input_saved_model_dir': './saved_model_fp32',
        'input_saved_model_tags': [],
        'input_saved_model_signature_key': 'serving_default',
        'use_dynamic_shape': np.bool_(False),
        'dynamic_shape_profile_strategy': 'Range',
        'max_workspace_size_bytes': np.int64(1 << 30), # 1GB
        'precision_mode': 'FP32',
        'minimum_segment_size': np.int32(3),
        'maximum_cached_engines': np.int32(1),
        'use_calibration': np.bool_(False),
        'allow_build_at_runtime': np.bool_(True),
        'conversion_params': 'params_fp32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: FP16 conversion with different parameters.
    input_dict_2 = {
        'input_saved_model_dir': './saved_model_fp16',
        'input_saved_model_tags': [],
        'input_saved_model_signature_key': 'predict',
        'use_dynamic_shape': np.bool_(False),
        'dynamic_shape_profile_strategy': 'Optimal',
        'max_workspace_size_bytes': np.int64(2 << 30), # 2GB
        'precision_mode': 'FP16',
        'minimum_segment_size': np.int32(5),
        'maximum_cached_engines': np.int32(16),
        'use_calibration': np.bool_(False),
        'allow_build_at_runtime': np.bool_(True),
        'conversion_params': 'params_fp16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: INT8 conversion with calibration enabled.
    input_dict_3 = {
        'input_saved_model_dir': './saved_model_int8',
        'input_saved_model_tags': [],
        'input_saved_model_signature_key': 'serving_default',
        'use_dynamic_shape': np.bool_(False),
        'dynamic_shape_profile_strategy': 'Range',
        'max_workspace_size_bytes': np.int64(1 << 30), # 1GB
        'precision_mode': 'INT8',
        'minimum_segment_size': np.int32(3),
        'maximum_cached_engines': np.int32(1),
        'use_calibration': np.bool_(True),
        'allow_build_at_runtime': np.bool_(True),
        'conversion_params': 'params_int8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    return list_of_inputs

generated_inputs["tf.experimental.tensorrt.Converter"] = tf_experimental_tensorrt_converter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.tensorrt.Converter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.tensorrt.Converter'.")

check_valid('tf.experimental.tensorrt.Converter', generated_inputs['tf.experimental.tensorrt.Converter'], lib="tf", suffix=0)

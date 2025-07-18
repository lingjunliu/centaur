
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_experimental_tensorrt_converter_inputs():
    """
    Generates a list of valid inputs for the tf.experimental.tensorrt.Converter API.
    NOTE: This API requires a specific build of TensorFlow with TensorRT support.
    The execution environment may raise a RuntimeError if this support is
    missing. The generated inputs are syntactically valid for the API itself.
    """
    list_of_inputs = []

    base_input = {
        'input_saved_model_dir': 'my_saved_model_dir',
        'input_saved_model_tags': [],
        'input_saved_model_signature_key': 'serving_default',
        'use_dynamic_shape': False,
        'dynamic_shape_profile_strategy': 'Range',
        'max_workspace_size_bytes': 1073741824,
        'precision_mode': 'FP32',
        'minimum_segment_size': 3,
        'maximum_cached_engines': 1,
        'use_calibration': True,
        'allow_build_at_runtime': True,
        'conversion_params': 'placeholder_for_conversion_params_object'
    }

    # Input 1: Basic FP32 conversion
    input_1 = copy.deepcopy(base_input)
    list_of_inputs.append(input_1)

    # Input 2: Basic FP16 conversion
    input_2 = copy.deepcopy(base_input)
    input_2['precision_mode'] = 'FP16'
    input_2['input_saved_model_dir'] = 'my_fp16_model'
    list_of_inputs.append(input_2)

    # Input 3: FP16 with pre-built engine caching
    input_3 = copy.deepcopy(base_input)
    input_3['precision_mode'] = 'FP16'
    input_3['maximum_cached_engines'] = 16
    input_3['input_saved_model_dir'] = 'my_fp16_cached_model'
    list_of_inputs.append(input_3)

    # Input 4: INT8 conversion with calibration
    input_4 = copy.deepcopy(base_input)
    input_4['precision_mode'] = 'INT8'
    input_4['use_calibration'] = True
    input_4['maximum_cached_engines'] = 1
    input_4['input_saved_model_dir'] = 'my_int8_model'
    list_of_inputs.append(input_4)

    # Input 5: Dynamic Shape with 'Range' profile strategy
    input_5 = copy.deepcopy(base_input)
    input_5['use_dynamic_shape'] = True
    input_5['dynamic_shape_profile_strategy'] = 'Range'
    input_5['input_saved_model_dir'] = 'my_dynamic_range_model'
    list_of_inputs.append(input_5)

    # Input 6: Dynamic Shape with 'Optimal' profile strategy
    input_6 = copy.deepcopy(base_input)
    input_6['use_dynamic_shape'] = True
    input_6['dynamic_shape_profile_strategy'] = 'Optimal'
    input_6['input_saved_model_dir'] = 'my_dynamic_optimal_model'
    list_of_inputs.append(input_6)

    # Input 7: Custom parameters
    input_7 = copy.deepcopy(base_input)
    input_7['max_workspace_size_bytes'] = 8589934592
    input_7['minimum_segment_size'] = 5
    input_7['allow_build_at_runtime'] = False
    input_7['input_saved_model_dir'] = 'my_custom_config_model'
    list_of_inputs.append(input_7)

    # Input 8: INT8 conversion without calibration
    input_8 = copy.deepcopy(base_input)
    input_8['precision_mode'] = 'INT8'
    input_8['use_calibration'] = False
    input_8['input_saved_model_dir'] = 'my_int8_no_calib_model'
    list_of_inputs.append(input_8)

    # Input 9: Mixed non-default settings
    input_9 = copy.deepcopy(base_input)
    input_9['max_workspace_size_bytes'] = 2147483648
    input_9['minimum_segment_size'] = 2
    input_9['maximum_cached_engines'] = 4
    input_9['precision_mode'] = 'FP16'
    input_9['input_saved_model_dir'] = 'my_mixed_settings_model'
    list_of_inputs.append(input_9)

    # Input 10: All parameters set to non-default values where applicable
    input_10 = {
        'input_saved_model_dir': 'all_params_model',
        'input_saved_model_tags': [],
        'input_saved_model_signature_key': 'another_key',
        'use_dynamic_shape': True,
        'dynamic_shape_profile_strategy': 'Optimal',
        'max_workspace_size_bytes': 4294967296,
        'precision_mode': 'FP16',
        'minimum_segment_size': 10,
        'maximum_cached_engines': 10,
        'use_calibration': False,
        'allow_build_at_runtime': False,
        'conversion_params': 'another_placeholder_string'
    }
    list_of_inputs.append(input_10)

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

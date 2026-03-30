
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_tensorrt_converter_inputs():
    list_of_inputs = []

    # Input 1, minimal valid input
    input_dict = {
        'input_saved_model_dir': 'test_dir',
        'input_saved_model_tags': ['serve'],
        'input_saved_model_signature_key': 'serving_default',
        'use_dynamic_shape': False,
        'dynamic_shape_profile_strategy': 'Range',
        'max_workspace_size_bytes': 1000000000,
        'precision_mode': 'FP32',
        'minimum_segment_size': 3,
        'maximum_cached_engines': 1,
        'use_calibration': True,
        'allow_build_at_runtime': True,
        'conversion_params': 'params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, different tags and signature
    input_dict = {
        'input_saved_model_dir': 'another_dir',
        'input_saved_model_tags': ['gpu', 'serving'],
        'input_saved_model_signature_key': 'my_signature',
        'use_dynamic_shape': True,
        'dynamic_shape_profile_strategy': 'Optimal',
        'max_workspace_size_bytes': 2000000000,
        'precision_mode': 'FP16',
        'minimum_segment_size': 5,
        'maximum_cached_engines': 4,
        'use_calibration': False,
        'allow_build_at_runtime': False,
        'conversion_params': 'params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, INT8 precision
    input_dict = {
        'input_saved_model_dir': 'int8_dir',
        'input_saved_model_tags': ['int8'],
        'input_saved_model_signature_key': 'int8_sig',
        'use_dynamic_shape': False,
        'dynamic_shape_profile_strategy': 'Range+Optimal',
        'max_workspace_size_bytes': 500000000,
        'precision_mode': 'INT8',
        'minimum_segment_size': 2,
        'maximum_cached_engines': 1,
        'use_calibration': True,
        'allow_build_at_runtime': True,
        'conversion_params': 'params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, dynamic shape with different strategy
    input_dict = {
        'input_saved_model_dir': 'dynamic_dir',
        'input_saved_model_tags': ['dynamic'],
        'input_saved_model_signature_key': 'dynamic_sig',
        'use_dynamic_shape': True,
        'dynamic_shape_profile_strategy': 'Range',
        'max_workspace_size_bytes': 3000000000,
        'precision_mode': 'FP32',
        'minimum_segment_size': 7,
        'maximum_cached_engines': 8,
        'use_calibration': True,
        'allow_build_at_runtime': False,
        'conversion_params': 'params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, small workspace size
    input_dict = {
        'input_saved_model_dir': 'small_dir',
        'input_saved_model_tags': ['small'],
        'input_saved_model_signature_key': 'small_sig',
        'use_dynamic_shape': False,
        'dynamic_shape_profile_strategy': 'Optimal',
        'max_workspace_size_bytes': 100000,
        'precision_mode': 'FP16',
        'minimum_segment_size': 1,
        'maximum_cached_engines': 2,
        'use_calibration': False,
        'allow_build_at_runtime': True,
        'conversion_params': 'params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, no calibration
    input_dict = {
        'input_saved_model_dir': 'no_calib_dir',
        'input_saved_model_tags': ['no_calib'],
        'input_saved_model_signature_key': 'no_calib_sig',
        'use_dynamic_shape': True,
        'dynamic_shape_profile_strategy': 'Range+Optimal',
        'max_workspace_size_bytes': 4000000000,
        'precision_mode': 'INT8',
        'minimum_segment_size': 4,
        'maximum_cached_engines': 1,
        'use_calibration': False,
        'allow_build_at_runtime': False,
        'conversion_params': 'params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: custom tag and signature
    input_dict = {
        'input_saved_model_dir': 'custom_model',
        'input_saved_model_tags': ['custom_tag'],
        'input_saved_model_signature_key': 'custom_signature_key',
        'use_dynamic_shape': True,
        'dynamic_shape_profile_strategy': 'Range',
        'max_workspace_size_bytes': 2147483648,
        'precision_mode': 'FP32',
        'minimum_segment_size': 6,
        'maximum_cached_engines': 5,
        'use_calibration': True,
        'allow_build_at_runtime': True,
         'conversion_params': 'params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different precision mode
    input_dict = {
        'input_saved_model_dir': 'precision_test',
        'input_saved_model_tags': ['precision'],
        'input_saved_model_signature_key': 'precision_sig',
        'use_dynamic_shape': False,
        'dynamic_shape_profile_strategy': 'Optimal',
        'max_workspace_size_bytes': 8589934592,
        'precision_mode': 'FP16',
        'minimum_segment_size': 8,
        'maximum_cached_engines': 10,
        'use_calibration': False,
        'allow_build_at_runtime': False,
        'conversion_params': 'params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different segment size and cached engines
    input_dict = {
        'input_saved_model_dir': 'segment_engine',
        'input_saved_model_tags': ['segment', 'engine'],
        'input_saved_model_signature_key': 'segment_engine_sig',
        'use_dynamic_shape': True,
        'dynamic_shape_profile_strategy': 'Range+Optimal',
        'max_workspace_size_bytes': 4294967296,
        'precision_mode': 'INT8',
        'minimum_segment_size': 10,
        'maximum_cached_engines': 20,
        'use_calibration': True,
        'allow_build_at_runtime': True,
        'conversion_params': 'params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: allow_build_at_runtime = False
    input_dict = {
        'input_saved_model_dir': 'runtime_test',
        'input_saved_model_tags': ['runtime'],
        'input_saved_model_signature_key': 'runtime_sig',
        'use_dynamic_shape': False,
        'dynamic_shape_profile_strategy': 'Range',
        'max_workspace_size_bytes': 17179869184,
        'precision_mode': 'FP32',
        'minimum_segment_size': 12,
        'maximum_cached_engines': 30,
        'use_calibration': False,
        'allow_build_at_runtime': False,
        'conversion_params': 'params'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
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


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.tensorrt.Converter', generated_inputs['tf.experimental.tensorrt.Converter'], lib="tf", suffix=0)

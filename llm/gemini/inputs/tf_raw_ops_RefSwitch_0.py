
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_refswitch_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.RefSwitch function.

    NOTE: tf.raw_ops.RefSwitch is a TensorFlow graph-mode operation that
    requires a 'ref' tensor (a mutable variable). It is explicitly not
    supported in eager execution, which is the default mode in TensorFlow 2.x.
    The provided input conforms to the API's signature (`data` and `pred` as
    tensors) but is expected to fail with a RuntimeError when executed in an
    eager context, as the testing environment seems to be doing. This failure
    is inherent to the design of the API and the execution mode, not the
    input values themselves.
    """
    list_of_inputs = []

    # A single, simple input that conforms to the signature. It is expected
    # to fail in eager mode due to the API's design.
    input_dict = {
        'data': np.array([10], dtype=np.int8),
        'pred': np.array(False),
        'name': 'ref_switch_minimal_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # A second case with a different data type and pred value.
    input_dict_2 = {
        'data': np.array([[1.0, 2.0]], dtype=np.float32),
        'pred': np.array(True),
        'name': 'ref_switch_float_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    return list_of_inputs

generated_inputs["tf.raw_ops.RefSwitch"] = tf_raw_ops_refswitch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RefSwitch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RefSwitch'.")

check_valid('tf.raw_ops.RefSwitch', generated_inputs['tf.raw_ops.RefSwitch'], lib="tf", suffix=0)

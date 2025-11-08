
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_LoopCond_inputs():
    list_of_inputs = []

    inp = np.array(True, dtype=np.bool_)
    input_dict = {"name": "loopcond_true_scalar_array", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.array(False, dtype=np.bool_)
    input_dict = {"name": "loopcond_false_scalar_array", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.bool_(True)
    input_dict = {"name": "loopcond_true_numpy_scalar", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.bool_(False)
    input_dict = {"name": "loopcond_false_numpy_scalar", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.array(1, dtype=np.bool_)
    input_dict = {"name": "loopcond_from_int_one", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.array(0, dtype=np.bool_)
    input_dict = {"name": "loopcond_from_int_zero", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.asarray(True, dtype=bool)
    input_dict = {"name": "loopcond_asarray_true", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.asarray(False, dtype=bool)
    input_dict = {"name": "loopcond_asarray_false", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.full((), True, dtype=np.bool_)
    input_dict = {"name": "loopcond_full_true", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.full((), False, dtype=np.bool_)
    input_dict = {"name": "loopcond_full_false", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.array(True, dtype='?')
    input_dict = {"name": "loopcond_dtype_questionmark_true", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.array(np.bool_(True))
    input_dict = {"name": "loopcond_wrapped_numpy_bool_true", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.LoopCond"] = tf_raw_ops_LoopCond_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.LoopCond' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LoopCond'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.LoopCond', generated_inputs['tf.raw_ops.LoopCond'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.compat.v1.enable_resource_variables()
tf.compat.v1.enable_eager_execution()
tf.random.set_seed(42)

def tf_raw_ops_apply_ftrl_v2_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    linear = np.array([0.4], dtype=np.float32)
    grad = np.array([0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.02, dtype=np.float32)
    l2 = np.array(0.03, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "ftrl_update_1"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([1.0], dtype=np.float64)
    accum = np.array([0.1], dtype=np.float64)
    linear = np.array([0.4], dtype=np.float64)
    grad = np.array([0.7], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    l1 = np.array(0.02, dtype=np.float64)
    l2 = np.array(0.03, dtype=np.float64)
    l2_shrinkage = np.array(0.001, dtype=np.float64)
    lr_power = np.array(-0.5, dtype=np.float64)
    use_locking = True
    multiply_linear_by_lr = True
    name = "ftrl_update_2"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([1], dtype=np.int32)
    accum = np.array([0], dtype=np.int32)
    linear = np.array([0], dtype=np.int32)
    grad = np.array([1], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(1, dtype=np.int32)
    l2 = np.array(1, dtype=np.int32)
    l2_shrinkage = np.array(0, dtype=np.int32)
    lr_power = np.array(0, dtype=np.int32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "ftrl_update_3"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    var = np.array([-1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    linear = np.array([0.4], dtype=np.float32)
    grad = np.array([0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.02, dtype=np.float32)
    l2 = np.array(0.03, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "ftrl_update_4"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([-0.1], dtype=np.float32)
    linear = np.array([0.4], dtype=np.float32)
    grad = np.array([0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.02, dtype=np.float32)
    l2 = np.array(0.03, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "ftrl_update_5"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    linear = np.array([-0.4], dtype=np.float32)
    grad = np.array([0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.02, dtype=np.float32)
    l2 = np.array(0.03, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "ftrl_update_6"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    linear = np.array([0.4], dtype=np.float32)
    grad = np.array([-0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.02, dtype=np.float32)
    l2 = np.array(0.03, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "ftrl_update_7"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    linear = np.array([0.4], dtype=np.float32)
    grad = np.array([0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.02, dtype=np.float32)
    l2 = np.array(0.03, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "ftrl_update_8"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    linear = np.array([0.4], dtype=np.float32)
    grad = np.array([0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.02, dtype=np.float32)
    l2 = np.array(0.03, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "ftrl_update_9"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    var = np.array([1.0], dtype=np.float32)
    accum = np.array([0.1], dtype=np.float32)
    linear = np.array([0.4], dtype=np.float32)
    grad = np.array([0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.02, dtype=np.float32)
    l2 = np.array(0.03, dtype=np.float32)
    l2_shrinkage = np.array(0.001, dtype=np.float32)
    lr_power = np.array(-0.5, dtype=np.float32)
    use_locking = False
    multiply_linear_by_lr = False
    name = "ftrl_update_10"

    input_dict = {
        "var": var,
        "accum": accum,
        "linear": linear,
        "grad": grad,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "l2_shrinkage": l2_shrinkage,
        "lr_power": lr_power,
        "use_locking": use_locking,
        "multiply_linear_by_lr": multiply_linear_by_lr,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyFtrlV2"] = tf_raw_ops_apply_ftrl_v2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyFtrlV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyFtrlV2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ApplyFtrlV2', generated_inputs['tf.raw_ops.ApplyFtrlV2'], lib="tf", suffix=0)

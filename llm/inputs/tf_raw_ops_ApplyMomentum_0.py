
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_momentum_inputs():
    list_of_inputs = []

    # Input 1: Basic test case
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_1"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    for k in input_dict:
      if k in ['lr', 'grad', 'momentum']:
        input_dict[k] = tf.convert_to_tensor(input_dict[k])
    input_dict["var"] = tf.Variable(input_dict["var"])
    input_dict["accum"] = tf.Variable(input_dict["accum"])

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Test with locking
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = True
    use_nesterov = False
    name = "momentum_update_2"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    for k in input_dict:
      if k in ['lr', 'grad', 'momentum']:
        input_dict[k] = tf.convert_to_tensor(input_dict[k])
    input_dict["var"] = tf.Variable(input_dict["var"])
    input_dict["accum"] = tf.Variable(input_dict["accum"])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Test with Nesterov momentum
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = True
    name = "momentum_update_3"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    for k in input_dict:
      if k in ['lr', 'grad', 'momentum']:
        input_dict[k] = tf.convert_to_tensor(input_dict[k])
    input_dict["var"] = tf.Variable(input_dict["var"])
    input_dict["accum"] = tf.Variable(input_dict["accum"])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Test with a different dtype (float64)
    var = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    accum = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    grad = np.array([0.4, 0.5, 0.6], dtype=np.float64)
    momentum = np.array(0.9, dtype=np.float64)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_4"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    for k in input_dict:
      if k in ['lr', 'grad', 'momentum']:
        input_dict[k] = tf.convert_to_tensor(input_dict[k])
    input_dict["var"] = tf.Variable(input_dict["var"])
    input_dict["accum"] = tf.Variable(input_dict["accum"])
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Test with negative values
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    accum = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    lr = np.array(-0.01, dtype=np.float32)
    grad = np.array([-0.4, -0.5, -0.6], dtype=np.float32)
    momentum = np.array(-0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_5"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    for k in input_dict:
      if k in ['lr', 'grad', 'momentum']:
        input_dict[k] = tf.convert_to_tensor(input_dict[k])
    input_dict["var"] = tf.Variable(input_dict["var"])
    input_dict["accum"] = tf.Variable(input_dict["accum"])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Test with zero values
    var = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lr = np.array(0.0, dtype=np.float32)
    grad = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    momentum = np.array(0.0, dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_6"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    for k in input_dict:
      if k in ['lr', 'grad', 'momentum']:
        input_dict[k] = tf.convert_to_tensor(input_dict[k])
    input_dict["var"] = tf.Variable(input_dict["var"])
    input_dict["accum"] = tf.Variable(input_dict["accum"])
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Test with small values
    var = np.array([1e-2, 2e-2, 3e-2], dtype=np.float32)
    accum = np.array([1e-7, 2e-7, 3e-7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([4e-7, 5e-7, 6e-7], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_7"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    for k in input_dict:
      if k in ['lr', 'grad', 'momentum']:
        input_dict[k] = tf.convert_to_tensor(input_dict[k])
    input_dict["var"] = tf.Variable(input_dict["var"])
    input_dict["accum"] = tf.Variable(input_dict["accum"])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Test with different shapes (2D arrays)
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    accum = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_8"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    for k in input_dict:
      if k in ['lr', 'grad', 'momentum']:
        input_dict[k] = tf.convert_to_tensor(input_dict[k])
    input_dict["var"] = tf.Variable(input_dict["var"])
    input_dict["accum"] = tf.Variable(input_dict["accum"])
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Test with different shapes (3D arrays)
    var = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    accum = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    grad = np.array([[[0.4, 0.5], [0.6, 0.7]], [[0.8, 0.9], [1.0, 1.1]]], dtype=np.float32)
    momentum = np.array(0.9, dtype=np.float32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_9"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    for k in input_dict:
      if k in ['lr', 'grad', 'momentum']:
        input_dict[k] = tf.convert_to_tensor(input_dict[k])
    input_dict["var"] = tf.Variable(input_dict["var"])
    input_dict["accum"] = tf.Variable(input_dict["accum"])
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Test with int32 data type
    var = np.array([1, 2, 3], dtype=np.int32)
    accum = np.array([1, 2, 3], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    grad = np.array([4, 5, 6], dtype=np.int32)
    momentum = np.array(1, dtype=np.int32)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_10"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    for k in input_dict:
      if k in ['lr', 'grad', 'momentum']:
        input_dict[k] = tf.convert_to_tensor(input_dict[k])
    input_dict["var"] = tf.Variable(input_dict["var"])
    input_dict["accum"] = tf.Variable(input_dict["accum"])
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: uint8
    var = np.array([1, 2, 3], dtype=np.uint8)
    accum = np.array([1, 2, 3], dtype=np.uint8)
    lr = np.array(1, dtype=np.uint8)
    grad = np.array([4, 5, 6], dtype=np.uint8)
    momentum = np.array(1, dtype=np.uint8)
    use_locking = False
    use_nesterov = False
    name = "momentum_update_11"

    input_dict = {
        "var": var,
        "accum": accum,
        "lr": lr,
        "grad": grad,
        "momentum": momentum,
        "use_locking": use_locking,
        "use_nesterov": use_nesterov,
        "name": name
    }
    for k in input_dict:
      if k in ['lr', 'grad', 'momentum']:
        input_dict[k] = tf.convert_to_tensor(input_dict[k])
    input_dict["var"] = tf.Variable(input_dict["var"])
    input_dict["accum"] = tf.Variable(input_dict["accum"])
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyMomentum"] = tf_raw_ops_apply_momentum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyMomentum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyMomentum'.")

check_valid('tf.raw_ops.ApplyMomentum', generated_inputs['tf.raw_ops.ApplyMomentum'], lib="tf", suffix=0)

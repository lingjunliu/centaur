
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_apply_adagrad_da_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    gradient_accumulator = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    gradient_squared_accumulator = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    global_step = np.array(10, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_1"

    input_dict = {
        "var": tf.Variable(var).read_value(),
        "gradient_accumulator": tf.Variable(gradient_accumulator).read_value(),
        "gradient_squared_accumulator": tf.Variable(gradient_squared_accumulator).read_value(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float32),
        "global_step": tf.convert_to_tensor(global_step, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    lr = np.array(0.005, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.01, dtype=np.float32)
    global_step = np.array(20, dtype=np.int64)
    use_locking = True
    name = "adagrad_da_2"

    input_dict = {
        "var": tf.Variable(var).read_value(),
        "gradient_accumulator": tf.Variable(gradient_accumulator).read_value(),
        "gradient_squared_accumulator": tf.Variable(gradient_squared_accumulator).read_value(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float32),
        "global_step": tf.convert_to_tensor(global_step, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([1, 2, 3], dtype=np.int32)
    gradient_accumulator = np.array([0, 0, 0], dtype=np.int32)
    gradient_squared_accumulator = np.array([0, 0, 0], dtype=np.int32)
    grad = np.array([1, 1, 1], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    global_step = np.array(1, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_3"

    input_dict = {
        "var": tf.Variable(var).read_value(),
        "gradient_accumulator": tf.Variable(gradient_accumulator).read_value(),
        "gradient_squared_accumulator": tf.Variable(gradient_squared_accumulator).read_value(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.int32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.int32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.int32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.int32),
        "global_step": tf.convert_to_tensor(global_step, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    var = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    gradient_accumulator = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    gradient_squared_accumulator = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    grad = np.array([-0.5, -0.6, -0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    global_step = np.array(10, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_4"

    input_dict = {
        "var": tf.Variable(var).read_value(),
        "gradient_accumulator": tf.Variable(gradient_accumulator).read_value(),
        "gradient_squared_accumulator": tf.Variable(gradient_squared_accumulator).read_value(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float32),
        "global_step": tf.convert_to_tensor(global_step, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = np.array([1.0], dtype=np.float64)
    gradient_accumulator = np.array([0.1], dtype=np.float64)
    gradient_squared_accumulator = np.array([0.01], dtype=np.float64)
    grad = np.array([0.5], dtype=np.float64)
    lr = np.array(0.01, dtype=np.float64)
    l1 = np.array(0.0, dtype=np.float64)
    l2 = np.array(0.0, dtype=np.float64)
    global_step = np.array(10, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_5"

    input_dict = {
        "var": tf.Variable(var).read_value(),
        "gradient_accumulator": tf.Variable(gradient_accumulator).read_value(),
        "gradient_squared_accumulator": tf.Variable(gradient_squared_accumulator).read_value(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float64),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float64),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float64),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float64),
        "global_step": tf.convert_to_tensor(global_step, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array(1, dtype=np.int64)
    gradient_accumulator = np.array(0, dtype=np.int64)
    gradient_squared_accumulator = np.array(0, dtype=np.int64)
    grad = np.array(1, dtype=np.int64)
    lr = np.array(1, dtype=np.int64)
    l1 = np.array(0, dtype=np.int64)
    l2 = np.array(0, dtype=np.int64)
    global_step = np.array(1, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_6"

    input_dict = {
        "var": tf.Variable(var).read_value(),
        "gradient_accumulator": tf.Variable(gradient_accumulator).read_value(),
        "gradient_squared_accumulator": tf.Variable(gradient_squared_accumulator).read_value(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.int64),
        "lr": tf.convert_to_tensor(lr, dtype=tf.int64),
        "l1": tf.convert_to_tensor(l1, dtype=tf.int64),
        "l2": tf.convert_to_tensor(l2, dtype=tf.int64),
        "global_step": tf.convert_to_tensor(global_step, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    gradient_accumulator = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    gradient_squared_accumulator = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    grad = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    global_step = np.array(10, dtype=np.int64)
    use_locking = True
    name = "adagrad_da_7"

    input_dict = {
        "var": tf.Variable(var).read_value(),
        "gradient_accumulator": tf.Variable(gradient_accumulator).read_value(),
        "gradient_squared_accumulator": tf.Variable(gradient_squared_accumulator).read_value(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float32),
        "global_step": tf.convert_to_tensor(global_step, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    var = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    gradient_accumulator = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[[0.01, 0.02], [0.03, 0.04]], [[0.05, 0.06], [0.07, 0.08]]], dtype=np.float32)
    grad = np.array([[[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]], dtype=np.float32)
    lr = np.array(0.005, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.01, dtype=np.float32)
    global_step = np.array(20, dtype=np.int64)
    use_locking = True
    name = "adagrad_da_8"

    input_dict = {
        "var": tf.Variable(var).read_value(),
        "gradient_accumulator": tf.Variable(gradient_accumulator).read_value(),
        "gradient_squared_accumulator": tf.Variable(gradient_squared_accumulator).read_value(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float32),
        "global_step": tf.convert_to_tensor(global_step, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = np.array([[1, 2], [3, 4]], dtype=np.int32)
    gradient_accumulator = np.array([[0, 0], [0, 0]], dtype=np.int32)
    gradient_squared_accumulator = np.array([[0, 0], [0, 0]], dtype=np.int32)
    grad = np.array([[1, 1], [1, 1]], dtype=np.int32)
    lr = np.array(1, dtype=np.int32)
    l1 = np.array(0, dtype=np.int32)
    l2 = np.array(0, dtype=np.int32)
    global_step = np.array(1, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_9"

    input_dict = {
        "var": tf.Variable(var).read_value(),
        "gradient_accumulator": tf.Variable(gradient_accumulator).read_value(),
        "gradient_squared_accumulator": tf.Variable(gradient_squared_accumulator).read_value(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.int32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.int32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.int32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.int32),
        "global_step": tf.convert_to_tensor(global_step, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    var = np.array([[-1.0, -2.0],[-3.0, -4.0]], dtype=np.float32)
    gradient_accumulator = np.array([[-0.1, -0.2],[-0.3, -0.4]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[0.01, 0.02],[0.03, 0.04]], dtype=np.float32)
    grad = np.array([[-0.5, -0.6],[-0.7, -0.8]], dtype=np.float32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    global_step = np.array(10, dtype=np.int64)
    use_locking = False
    name = "adagrad_da_10"

    input_dict = {
        "var": tf.Variable(var).read_value(),
        "gradient_accumulator": tf.Variable(gradient_accumulator).read_value(),
        "gradient_squared_accumulator": tf.Variable(gradient_squared_accumulator).read_value(),
        "grad": tf.convert_to_tensor(grad, dtype=tf.float32),
        "lr": tf.convert_to_tensor(lr, dtype=tf.float32),
        "l1": tf.convert_to_tensor(l1, dtype=tf.float32),
        "l2": tf.convert_to_tensor(l2, dtype=tf.float32),
        "global_step": tf.convert_to_tensor(global_step, dtype=tf.int64),
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApplyAdagradDA"] = tf_raw_ops_apply_adagrad_da_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApplyAdagradDA' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdagradDA'.")

check_valid('tf.raw_ops.ApplyAdagradDA', generated_inputs['tf.raw_ops.ApplyAdagradDA'], lib="tf", suffix=0)

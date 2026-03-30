
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseApplyAdagradDA_inputs():
    list_of_inputs = []

    # Input 1
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    global_step = np.array(1, dtype=np.int64)
    use_locking = False
    name = "sparse_apply_adagrad_da_1"

    var_tensor = tf.Variable(var, dtype=tf.float32)
    gradient_accumulator_tensor = tf.Variable(gradient_accumulator, dtype=tf.float32)
    gradient_squared_accumulator_tensor = tf.Variable(gradient_squared_accumulator, dtype=tf.float32)
    grad_tensor = tf.convert_to_tensor(grad, dtype=tf.float32)
    indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
    lr_tensor = tf.convert_to_tensor(lr, dtype=tf.float32)
    l1_tensor = tf.convert_to_tensor(l1, dtype=tf.float32)
    l2_tensor = tf.convert_to_tensor(l2, dtype=tf.float32)
    global_step_tensor = tf.convert_to_tensor(global_step, dtype=tf.int64)

    input_dict = {
        "var": var_tensor,
        "gradient_accumulator": gradient_accumulator_tensor,
        "gradient_squared_accumulator": gradient_squared_accumulator_tensor,
        "grad": grad_tensor,
        "indices": indices_tensor,
        "lr": lr_tensor,
        "l1": l1_tensor,
        "l2": l2_tensor,
        "global_step": global_step_tensor,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0], dtype=np.int64)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    global_step = np.array(1, dtype=np.int64)
    use_locking = True
    name = "sparse_apply_adagrad_da_2"

    var_tensor = tf.Variable(var, dtype=tf.float32)
    gradient_accumulator_tensor = tf.Variable(gradient_accumulator, dtype=tf.float32)
    gradient_squared_accumulator_tensor = tf.Variable(gradient_squared_accumulator, dtype=tf.float32)
    grad_tensor = tf.convert_to_tensor(grad, dtype=tf.float32)
    indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int64)
    lr_tensor = tf.convert_to_tensor(lr, dtype=tf.float32)
    l1_tensor = tf.convert_to_tensor(l1, dtype=tf.float32)
    l2_tensor = tf.convert_to_tensor(l2, dtype=tf.float32)
    global_step_tensor = tf.convert_to_tensor(global_step, dtype=tf.int64)

    input_dict = {
        "var": var_tensor,
        "gradient_accumulator": gradient_accumulator_tensor,
        "gradient_squared_accumulator": gradient_squared_accumulator_tensor,
        "grad": grad_tensor,
        "indices": indices_tensor,
        "lr": lr_tensor,
        "l1": l1_tensor,
        "l2": l2_tensor,
        "global_step": global_step_tensor,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    grad = np.array([0.5, 0.6], dtype=np.float64)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float64)
    l1 = np.array(0.0, dtype=np.float64)
    l2 = np.array(0.0, dtype=np.float64)
    global_step = np.array(1, dtype=np.int64)
    use_locking = False
    name = "sparse_apply_adagrad_da_3"
    
    var_tensor = tf.Variable(var, dtype=tf.float64)
    gradient_accumulator_tensor = tf.Variable(gradient_accumulator, dtype=tf.float64)
    gradient_squared_accumulator_tensor = tf.Variable(gradient_squared_accumulator, dtype=tf.float64)
    grad_tensor = tf.convert_to_tensor(grad, dtype=tf.float64)
    indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
    lr_tensor = tf.convert_to_tensor(lr, dtype=tf.float64)
    l1_tensor = tf.convert_to_tensor(l1, dtype=tf.float64)
    l2_tensor = tf.convert_to_tensor(l2, dtype=tf.float64)
    global_step_tensor = tf.convert_to_tensor(global_step, dtype=tf.int64)

    input_dict = {
        "var": var_tensor,
        "gradient_accumulator": gradient_accumulator_tensor,
        "gradient_squared_accumulator": gradient_squared_accumulator_tensor,
        "grad": grad_tensor,
        "indices": indices_tensor,
        "lr": lr_tensor,
        "l1": l1_tensor,
        "l2": l2_tensor,
        "global_step": global_step_tensor,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0,1], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.01, dtype=np.float32)
    global_step = np.array(100, dtype=np.int64)
    use_locking = False
    name = "sparse_apply_adagrad_da_4"

    var_tensor = tf.Variable(var, dtype=tf.float32)
    gradient_accumulator_tensor = tf.Variable(gradient_accumulator, dtype=tf.float32)
    gradient_squared_accumulator_tensor = tf.Variable(gradient_squared_accumulator, dtype=tf.float32)
    grad_tensor = tf.convert_to_tensor(grad, dtype=tf.float32)
    indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
    lr_tensor = tf.convert_to_tensor(lr, dtype=tf.float32)
    l1_tensor = tf.convert_to_tensor(l1, dtype=tf.float32)
    l2_tensor = tf.convert_to_tensor(l2, dtype=tf.float32)
    global_step_tensor = tf.convert_to_tensor(global_step, dtype=tf.int64)

    input_dict = {
        "var": var_tensor,
        "gradient_accumulator": gradient_accumulator_tensor,
        "gradient_squared_accumulator": gradient_squared_accumulator_tensor,
        "grad": grad_tensor,
        "indices": indices_tensor,
        "lr": lr_tensor,
        "l1": l1_tensor,
        "l2": l2_tensor,
        "global_step": global_step_tensor,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([1], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.1, dtype=np.float32)
    l2 = np.array(0.01, dtype=np.float32)
    global_step = np.array(100, dtype=np.int64)
    use_locking = True
    name = "sparse_apply_adagrad_da_5"

    var_tensor = tf.Variable(var, dtype=tf.float32)
    gradient_accumulator_tensor = tf.Variable(gradient_accumulator, dtype=tf.float32)
    gradient_squared_accumulator_tensor = tf.Variable(gradient_squared_accumulator, dtype=tf.float32)
    grad_tensor = tf.convert_to_tensor(grad, dtype=tf.float32)
    indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
    lr_tensor = tf.convert_to_tensor(lr, dtype=tf.float32)
    l1_tensor = tf.convert_to_tensor(l1, dtype=tf.float32)
    l2_tensor = tf.convert_to_tensor(l2, dtype=tf.float32)
    global_step_tensor = tf.convert_to_tensor(global_step, dtype=tf.int64)

    input_dict = {
        "var": var_tensor,
        "gradient_accumulator": gradient_accumulator_tensor,
        "gradient_squared_accumulator": gradient_squared_accumulator_tensor,
        "grad": grad_tensor,
        "indices": indices_tensor,
        "lr": lr_tensor,
        "l1": l1_tensor,
        "l2": l2_tensor,
        "global_step": global_step_tensor,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(-0.1, dtype=np.float32)
    l2 = np.array(-0.01, dtype=np.float32)
    global_step = np.array(100, dtype=np.int64)
    use_locking = False
    name = "sparse_apply_adagrad_da_6"

    var_tensor = tf.Variable(var, dtype=tf.float32)
    gradient_accumulator_tensor = tf.Variable(gradient_accumulator, dtype=tf.float32)
    gradient_squared_accumulator_tensor = tf.Variable(gradient_squared_accumulator, dtype=tf.float32)
    grad_tensor = tf.convert_to_tensor(grad, dtype=tf.float32)
    indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
    lr_tensor = tf.convert_to_tensor(lr, dtype=tf.float32)
    l1_tensor = tf.convert_to_tensor(l1, dtype=tf.float32)
    l2_tensor = tf.convert_to_tensor(l2, dtype=tf.float32)
    global_step_tensor = tf.convert_to_tensor(global_step, dtype=tf.int64)


    input_dict = {
        "var": var_tensor,
        "gradient_accumulator": gradient_accumulator_tensor,
        "gradient_squared_accumulator": gradient_squared_accumulator_tensor,
        "grad": grad_tensor,
        "indices": indices_tensor,
        "lr": lr_tensor,
        "l1": l1_tensor,
        "l2": l2_tensor,
        "global_step": global_step_tensor,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0], dtype=np.int64)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(-0.1, dtype=np.float32)
    l2 = np.array(-0.01, dtype=np.float32)
    global_step = np.array(100, dtype=np.int64)
    use_locking = True
    name = "sparse_apply_adagrad_da_7"

    var_tensor = tf.Variable(var, dtype=tf.float32)
    gradient_accumulator_tensor = tf.Variable(gradient_accumulator, dtype=tf.float32)
    gradient_squared_accumulator_tensor = tf.Variable(gradient_squared_accumulator, dtype=tf.float32)
    grad_tensor = tf.convert_to_tensor(grad, dtype=tf.float32)
    indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int64)
    lr_tensor = tf.convert_to_tensor(lr, dtype=tf.float32)
    l1_tensor = tf.convert_to_tensor(l1, dtype=tf.float32)
    l2_tensor = tf.convert_to_tensor(l2, dtype=tf.float32)
    global_step_tensor = tf.convert_to_tensor(global_step, dtype=tf.int64)

    input_dict = {
        "var": var_tensor,
        "gradient_accumulator": gradient_accumulator_tensor,
        "gradient_squared_accumulator": gradient_squared_accumulator_tensor,
        "grad": grad_tensor,
        "indices": indices_tensor,
        "lr": lr_tensor,
        "l1": l1_tensor,
        "l2": l2_tensor,
        "global_step": global_step_tensor,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - different shape for var
    var = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    gradient_accumulator = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    gradient_squared_accumulator = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    grad = np.array([0.5], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    global_step = np.array(1, dtype=np.int64)
    use_locking = False
    name = "sparse_apply_adagrad_da_8"

    var_tensor = tf.Variable(var, dtype=tf.float32)
    gradient_accumulator_tensor = tf.Variable(gradient_accumulator, dtype=tf.float32)
    gradient_squared_accumulator_tensor = tf.Variable(gradient_squared_accumulator, dtype=tf.float32)
    grad_tensor = tf.convert_to_tensor(grad, dtype=tf.float32)
    indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
    lr_tensor = tf.convert_to_tensor(lr, dtype=tf.float32)
    l1_tensor = tf.convert_to_tensor(l1, dtype=tf.float32)
    l2_tensor = tf.convert_to_tensor(l2, dtype=tf.float32)
    global_step_tensor = tf.convert_to_tensor(global_step, dtype=tf.int64)

    input_dict = {
        "var": var_tensor,
        "gradient_accumulator": gradient_accumulator_tensor,
        "gradient_squared_accumulator": gradient_squared_accumulator_tensor,
        "grad": grad_tensor,
        "indices": indices_tensor,
        "lr": lr_tensor,
        "l1": l1_tensor,
        "l2": l2_tensor,
        "global_step": global_step_tensor,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0], dtype=np.int32)
    lr = np.array(-0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    global_step = np.array(1, dtype=np.int64)
    use_locking = False
    name = "sparse_apply_adagrad_da_9"

    var_tensor = tf.Variable(var, dtype=tf.float32)
    gradient_accumulator_tensor = tf.Variable(gradient_accumulator, dtype=tf.float32)
    gradient_squared_accumulator_tensor = tf.Variable(gradient_squared_accumulator, dtype=tf.float32)
    grad_tensor = tf.convert_to_tensor(grad, dtype=tf.float32)
    indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
    lr_tensor = tf.convert_to_tensor(lr, dtype=tf.float32)
    l1_tensor = tf.convert_to_tensor(l1, dtype=tf.float32)
    l2_tensor = tf.convert_to_tensor(l2, dtype=tf.float32)
    global_step_tensor = tf.convert_to_tensor(global_step, dtype=tf.int64)

    input_dict = {
        "var": var_tensor,
        "gradient_accumulator": gradient_accumulator_tensor,
        "gradient_squared_accumulator": gradient_squared_accumulator_tensor,
        "grad": grad_tensor,
        "indices": indices_tensor,
        "lr": lr_tensor,
        "l1": l1_tensor,
        "l2": l2_tensor,
        "global_step": global_step_tensor,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    var = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gradient_accumulator = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    gradient_squared_accumulator = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    grad = np.array([0.5, 0.6], dtype=np.float32)
    indices = np.array([0], dtype=np.int64)
    lr = np.array(0.01, dtype=np.float32)
    l1 = np.array(0.0, dtype=np.float32)
    l2 = np.array(0.0, dtype=np.float32)
    global_step = np.array(0, dtype=np.int64)
    use_locking = False
    name = "sparse_apply_adagrad_da_10"

    var_tensor = tf.Variable(var, dtype=tf.float32)
    gradient_accumulator_tensor = tf.Variable(gradient_accumulator, dtype=tf.float32)
    gradient_squared_accumulator_tensor = tf.Variable(gradient_squared_accumulator, dtype=tf.float32)
    grad_tensor = tf.convert_to_tensor(grad, dtype=tf.float32)
    indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int64)
    lr_tensor = tf.convert_to_tensor(lr, dtype=tf.float32)
    l1_tensor = tf.convert_to_tensor(l1, dtype=tf.float32)
    l2_tensor = tf.convert_to_tensor(l2, dtype=tf.float32)
    global_step_tensor = tf.convert_to_tensor(global_step, dtype=tf.int64)

    input_dict = {
        "var": var_tensor,
        "gradient_accumulator": gradient_accumulator_tensor,
        "gradient_squared_accumulator": gradient_squared_accumulator_tensor,
        "grad": grad_tensor,
        "indices": indices_tensor,
        "lr": lr_tensor,
        "l1": l1_tensor,
        "l2": l2_tensor,
        "global_step": global_step_tensor,
        "use_locking": use_locking,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
try:
    generated_inputs["tf.raw_ops.SparseApplyAdagradDA"] = tf_raw_ops_SparseApplyAdagradDA_inputs()
except Exception as e:
    print(f"Error generating inputs: {e}")

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SparseApplyAdagradDA' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseApplyAdagradDA'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SparseApplyAdagradDA', generated_inputs['tf.raw_ops.SparseApplyAdagradDA'], lib="tf", suffix=0)

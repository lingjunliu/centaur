
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_apply_adam_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ApplyAdam function.
    To satisfy both the API's requirement for mutable variables in eager mode
    and a testing harness that expects a .size attribute, the mutable inputs
    (var, m, v) are created as tf.Variable objects and then monkey-patched
    with a .size attribute. Other tensor inputs are standard numpy arrays.
    """
    list_of_inputs = []

    def create_input_dict(var_shape, np_dtype, use_locking, use_nesterov, name_suffix):
        
        # Create numpy arrays first
        var_np = np.random.randn(*var_shape).astype(np_dtype)
        m_np = np.random.randn(*var_shape).astype(np_dtype)
        # v (variance) must be non-negative for the sqrt operation in the algorithm.
        v_np = np.abs(np.random.randn(*var_shape)).astype(np_dtype)

        # Create tf.Variables as required by the API for mutable inputs
        var = tf.Variable(var_np)
        m = tf.Variable(m_np)
        v = tf.Variable(v_np)

        # Monkey-patch the .size attribute for the testing harness
        var.size = var_np.size
        m.size = m_np.size
        v.size = v_np.size

        # Other tensor inputs can be numpy arrays as per the prompt's preference
        grad = np.random.randn(*var_shape).astype(np_dtype)
        beta1_power = np.array(0.9**2, dtype=np_dtype)
        beta2_power = np.array(0.999**2, dtype=np_dtype)
        lr = np.array(0.001, dtype=np_dtype)
        beta1 = np.array(0.9, dtype=np_dtype)
        beta2 = np.array(0.999, dtype=np_dtype)
        epsilon = np.array(1e-7, dtype=np_dtype)

        return {
            'var': var,
            'm': m,
            'v': v,
            'beta1_power': beta1_power,
            'beta2_power': beta2_power,
            'lr': lr,
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'grad': grad,
            'use_locking': use_locking,
            'use_nesterov': use_nesterov,
            'name': f'apply_adam_{name_suffix}'
        }

    # Input 1: Basic case, float32, 2D
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        var_shape=(3, 3), np_dtype=np.float32, use_locking=False, use_nesterov=False, name_suffix="1"
    )))

    # Input 2: Nesterov enabled, float32, 2D
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        var_shape=(3, 3), np_dtype=np.float32, use_locking=False, use_nesterov=True, name_suffix="2"
    )))
    
    # Input 3: Locking enabled, float32, 2D
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        var_shape=(2, 4), np_dtype=np.float32, use_locking=True, use_nesterov=False, name_suffix="3"
    )))

    # Input 4: Nesterov and Locking enabled, float32, 2D
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        var_shape=(2, 4), np_dtype=np.float32, use_locking=True, use_nesterov=True, name_suffix="4"
    )))

    # Input 5: float64 dtype, 1D
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        var_shape=(10,), np_dtype=np.float64, use_locking=False, use_nesterov=False, name_suffix="5"
    )))

    # Input 6: Higher dimensions (3D), float32
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        var_shape=(2, 3, 4), np_dtype=np.float32, use_locking=False, use_nesterov=True, name_suffix="6"
    )))
    
    # Input 7: Zero gradient
    input_7 = create_input_dict(
        var_shape=(3, 2), np_dtype=np.float32, use_locking=False, use_nesterov=False, name_suffix="7"
    )
    input_7['grad'] = np.zeros((3, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: Zero-initialized m and v (typical first step)
    input_8 = create_input_dict(
        var_shape=(6,), np_dtype=np.float64, use_locking=True, use_nesterov=False, name_suffix="8"
    )
    m_np_8 = np.zeros(6, dtype=np.float64)
    v_np_8 = np.zeros(6, dtype=np.float64)
    m_8 = tf.Variable(m_np_8)
    v_8 = tf.Variable(v_np_8)
    m_8.size = m_np_8.size
    v_8.size = v_np_8.size
    input_8['m'] = m_8
    input_8['v'] = v_8
    input_8['beta1_power'] = np.array(0.9, dtype=np.float64) # t=1
    input_8['beta2_power'] = np.array(0.999, dtype=np.float64) # t=1
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: 4D tensor with different lr and epsilon
    input_9 = create_input_dict(
        var_shape=(1, 2, 2, 3), np_dtype=np.float32, use_locking=False, use_nesterov=False, name_suffix="9"
    )
    input_9['lr'] = np.array(0.1, dtype=np.float32)
    input_9['epsilon'] = np.array(1e-4, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_9))
    
    # Input 10: Negative values in var, m, and grad
    input_10 = create_input_dict(
        var_shape=(4,), np_dtype=np.float32, use_locking=False, use_nesterov=False, name_suffix="10"
    )
    var_np_10 = np.array([-1.0, -2.5, 3.0, -0.5], dtype=np.float32)
    m_np_10 = np.array([-0.1, 0.2, -0.05, 0.15], dtype=np.float32)
    var_10 = tf.Variable(var_np_10)
    m_10 = tf.Variable(m_np_10)
    var_10.size = var_np_10.size
    m_10.size = m_np_10.size
    input_10['var'] = var_10
    input_10['m'] = m_10
    input_10['grad'] = np.array([0.5, -1.0, -0.2, 0.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyAdam"] = tf_raw_ops_apply_adam_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyAdam' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyAdam'.")

check_valid('tf.raw_ops.ApplyAdam', generated_inputs['tf.raw_ops.ApplyAdam'], lib="tf", suffix=0)

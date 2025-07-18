
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

# This custom class is a workaround to satisfy conflicting requirements from
# the TensorFlow API and the testing framework.
# 1. The op requires a mutable ref type (a Variable), so we inherit from
#    ResourceVariable, the concrete implementation of tf.Variable.
# 2. The testing framework's analysis tools expect a `.size` attribute,
#    which we add as a property.
# 3. The framework uses `copy.deepcopy`, which fails on standard TF variables.
#    We implement a custom `__deepcopy__` to create a new, independent
#    variable with the same value, making it compatible.
class PatchedVariable(ResourceVariable):
    @property
    def size(self):
        if self.shape.is_fully_defined():
            return np.prod(self.shape.as_list())
        return tf.size(self).numpy()

    def __deepcopy__(self, memo):
        # Create a new PatchedVariable instance instead of deepcopying internal state.
        new_val = self.numpy()
        new_var = PatchedVariable(initial_value=new_val,
                                  dtype=self.dtype,
                                  name=self._shared_name + "_copy")
        memo[id(self)] = new_var
        return new_var

def get_tf_raw_ops_apply_power_sign_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.ApplyPowerSign function.
    """
    list_of_inputs = []

    # Helper function to create an input dictionary
    def create_input(var_dtype, var_shape, use_locking, name_suffix):
        var_init = np.random.randn(*var_shape).astype(var_dtype)
        m_init = np.random.randn(*var_shape).astype(var_dtype)

        # Use the patched variable class to meet all requirements
        var = PatchedVariable(initial_value=var_init, name=f"var_{name_suffix}")
        m = PatchedVariable(initial_value=m_init, name=f"m_{name_suffix}")

        grad = np.random.randn(*var_shape).astype(var_dtype)
        lr = np.array(0.001, dtype=var_dtype)
        logbase = np.array(np.e, dtype=var_dtype)
        sign_decay = np.array(0.99, dtype=var_dtype)
        beta = np.array(0.9, dtype=var_dtype)

        return {
            'var': var,
            'm': m,
            'lr': lr,
            'logbase': logbase,
            'sign_decay': sign_decay,
            'beta': beta,
            'grad': grad,
            'use_locking': use_locking,
            'name': f'ApplyPowerSign_{name_suffix}'
        }

    # Input 1: Basic float32, 1D
    list_of_inputs.append(copy.deepcopy(create_input(np.float32, (10,), False, 'float32_1d')))

    # Input 2: Basic float32, 2D with locking
    list_of_inputs.append(copy.deepcopy(create_input(np.float32, (3, 4), True, 'float32_2d_lock')))

    # Input 3: float64, 1D
    list_of_inputs.append(copy.deepcopy(create_input(np.float64, (5,), False, 'float64_1d')))

    # Input 4: float64, 3D
    list_of_inputs.append(copy.deepcopy(create_input(np.float64, (2, 3, 2), True, 'float64_3d_lock')))

    # Input 5: float32 with negative values in gradients
    input_5 = create_input(np.float32, (5, 5), False, 'float32_neg_grad')
    input_5['grad'] = -np.abs(input_5['grad'])
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: float32 with zero gradient
    input_6 = create_input(np.float32, (4, 2), True, 'float32_zero_grad')
    input_6['grad'] = np.zeros((4, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: half (float16)
    list_of_inputs.append(copy.deepcopy(create_input(np.float16, (8,), False, 'float16')))

    # Input 8: Large tensors, float32
    list_of_inputs.append(copy.deepcopy(create_input(np.float32, (50, 10), True, 'float32_large')))

    # Input 9: float32 with beta close to 1
    input_9 = create_input(np.float32, (7,), False, 'float32_high_beta')
    input_9['beta'] = np.array(0.999, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: float64 with a different logbase
    input_10 = create_input(np.float64, (6, 6), True, 'float64_logbase_10')
    input_10['logbase'] = np.array(10.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ApplyPowerSign"] = get_tf_raw_ops_apply_power_sign_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyPowerSign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyPowerSign'.")

check_valid('tf.raw_ops.ApplyPowerSign', generated_inputs['tf.raw_ops.ApplyPowerSign'], lib="tf", suffix=0)

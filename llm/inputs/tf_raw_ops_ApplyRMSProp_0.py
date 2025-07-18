
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

# MONKEY-PATCHING to satisfy the testing framework's API.
# The framework expects a numpy-like .size attribute on tensor-like objects,
# which tf.Variable and tf.EagerTensor (the type for tf.constant) do not have.
# This patch adds the required property to both classes.

# Patch tf.Variable for mutable inputs
if not hasattr(tf.Variable, 'size'):
    @property
    def __tf_variable_size_patch__(self):
        return tf.size(self).numpy()
    tf.Variable.size = __tf_variable_size_patch__

# Patch tf.EagerTensor for immutable tensor inputs
# EagerTensor is the type returned by tf.constant() in eager mode.
EagerTensor = type(tf.constant(1))
if not hasattr(EagerTensor, 'size'):
    @property
    def __eager_tensor_size_patch__(self):
        return tf.size(self).numpy()
    EagerTensor.size = __eager_tensor_size_patch__

def get_applyrmsprop_inputs():
  """
  Generates a list of valid inputs for the tf.raw_ops.ApplyRMSProp operation.
  This op requires mutable variables for var, ms, and mom to run in eager mode.
  We provide tf.Variable for these and tf.constant for other tensors, and patch
  them to be compatible with the testing framework.
  """
  list_of_inputs = []

  # Helper function to create input dictionaries with the correct TF types.
  def create_input_dict(var_np, ms_np, mom_np, lr_val, rho_val, momentum_val, epsilon_val, grad_np, use_locking, name):
    dtype = tf.as_dtype(var_np.dtype)
    return {
        'var': tf.Variable(var_np),
        'ms': tf.Variable(ms_np),
        'mom': tf.Variable(mom_np),
        'lr': tf.constant(lr_val, dtype=dtype),
        'rho': tf.constant(rho_val, dtype=dtype),
        'momentum': tf.constant(momentum_val, dtype=dtype),
        'epsilon': tf.constant(epsilon_val, dtype=dtype),
        'grad': tf.constant(grad_np, dtype=dtype),
        'use_locking': use_locking,
        'name': name
    }

  # Input 1: Basic float32, 1D
  dtype = np.float32
  list_of_inputs.append(create_input_dict(
      np.array([1.0, 2.0, 3.0], dtype=dtype),
      np.array([0.1, 0.2, 0.3], dtype=dtype),
      np.array([0.01, 0.02, 0.03], dtype=dtype),
      0.001, 0.9, 0.0, 1e-7,
      np.array([0.5, -0.5, 0.1], dtype=dtype),
      False, "basic_float32"
  ))

  # Input 2: Basic float64, 2D
  dtype = np.float64
  list_of_inputs.append(create_input_dict(
      np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype),
      np.array([[0.1, 0.2], [0.3, 0.4]], dtype=dtype),
      np.array([[0.01, 0.02], [0.03, 0.04]], dtype=dtype),
      0.01, 0.99, 0.5, 1e-8,
      np.array([[0.1, 0.2], [0.3, 0.4]], dtype=dtype),
      True, "basic_float64"
  ))

  # Input 3: Zero Gradient
  dtype = np.float32
  var3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
  list_of_inputs.append(create_input_dict(
      var3,
      np.array([[0.1, 0.2], [0.3, 0.4]], dtype=dtype),
      np.array([[0.01, 0.02], [0.03, 0.04]], dtype=dtype),
      0.001, 0.9, 0.9, 1e-7,
      np.zeros_like(var3),
      False, "zero_grad"
  ))

  # Input 4: complex64 type
  dtype = np.complex64
  list_of_inputs.append(create_input_dict(
      np.array([1+1j, 2-2j], dtype=dtype),
      np.array([0.1+0j, 0.2+0j], dtype=dtype),
      np.array([0.01+0.01j, 0.02-0.02j], dtype=dtype),
      0.001+0j, 0.9+0j, 0.0+0j, 1e-7+0j,
      np.array([0.5+0.1j, -0.5-0.2j], dtype=dtype),
      False, "complex64_type"
  ))

  # Input 5: bfloat16 type
  dtype = tf.bfloat16.as_numpy_dtype
  list_of_inputs.append(create_input_dict(
      np.array([[1.0, 2.0]], dtype=dtype),
      np.array([[0.1, 0.2]], dtype=dtype),
      np.array([[0.01, 0.02]], dtype=dtype),
      0.01, 0.99, 0.5, 1e-6,
      np.array([[0.1, 0.2]], dtype=dtype),
      False, "bfloat16_type"
  ))

  # Input 6: Single element tensor (0-D)
  dtype = np.float64
  list_of_inputs.append(create_input_dict(
      np.array(100.0, dtype=dtype),
      np.array(10.0, dtype=dtype),
      np.array(1.0, dtype=dtype),
      0.1, 0.9, 0.9, 1e-8,
      np.array(-5.0, dtype=dtype),
      False, "single_element_0d"
  ))

  # Input 7: 3D Tensor
  dtype = np.float32
  list_of_inputs.append(create_input_dict(
      np.random.rand(2, 2, 3).astype(dtype),
      np.abs(np.random.rand(2, 2, 3).astype(dtype)),
      np.random.rand(2, 2, 3).astype(dtype),
      0.001, 0.9, 0.0, 1e-7,
      np.random.rand(2, 2, 3).astype(dtype),
      True, "3d_tensor"
  ))
  
  return list_of_inputs

generated_inputs["tf.raw_ops.ApplyRMSProp"] = get_applyrmsprop_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ApplyRMSProp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApplyRMSProp'.")

check_valid('tf.raw_ops.ApplyRMSProp', generated_inputs['tf.raw_ops.ApplyRMSProp'], lib="tf", suffix=0)

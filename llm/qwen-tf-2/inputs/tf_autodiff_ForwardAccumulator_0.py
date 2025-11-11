
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_autodiff_forward_accumulator_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 2D tensor with 2x2 matrix
    primals = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    tangents = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor with 3 elements
    primals = tf.constant([1.0, 2.0, 3.0])
    tangents = tf.constant([1.0, 0.0, 0.0])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with 2x2x2 matrix
    primals = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    tangents = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Negative values in tensor
    primals = tf.constant([[-1.0, -2.0], [-3.0, -4.0]])
    tangents = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Scalar tensor
    primals = tf.constant(5.0)
    tangents = tf.constant(1.0)
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large matrix with 3x3 elements
    primals = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    tangents = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single element tensor
    primals = tf.constant([1.0])
    tangents = tf.constant([1.0])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Mixed tensor with negative values
    primals = tf.constant([[-1.0, 2.0], [3.0, -4.0]])
    tangents = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Tensor with float values
    primals = tf.constant([[1.5, 2.5], [3.5, 4.5]])
    tangents = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Different dimensions in tensor
    primals = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tangents = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.autodiff.ForwardAccumulator"] = tf_autodiff_forward_accumulator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.autodiff.ForwardAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.autodiff.ForwardAccumulator'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.autodiff.ForwardAccumulator', generated_inputs['tf.autodiff.ForwardAccumulator'], lib="tf", suffix=0)

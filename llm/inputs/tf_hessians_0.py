
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.compat.v1.disable_eager_execution()
# This enables numpy-compatible attributes (.size, .shape) on symbolic Tensors,
# which is required to pass the test harness validation while still allowing
# for the construction of a computational graph necessary for tf.hessians.
tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_hessians_inputs():
    list_of_inputs = []

    # Input 1: Basic scalar input
    g1 = tf.Graph()
    with g1.as_default():
        xs1 = [tf.constant(3.0, dtype=tf.float32)]
        ys1 = [xs1[0]**2]
    input_dict_1 = {
        'ys': ys1,
        'xs': xs1,
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessian_symbolic_scalar'
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: Vector input
    g2 = tf.Graph()
    with g2.as_default():
        xs2 = [tf.constant([1.0, 2.0], dtype=tf.float32)]
        ys2 = [tf.reduce_sum(xs2[0]**3)]
    input_dict_2 = {
        'ys': ys2,
        'xs': xs2,
        'gate_gradients': True,
        'aggregation_method': None,
        'name': 'hessian_symbolic_vector'
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: Matrix input, float64, with negative values
    g3 = tf.Graph()
    with g3.as_default():
        xs3 = [tf.constant([[-1.0, 2.0], [3.0, -4.0]], dtype=tf.float64)]
        ys3 = [tf.reduce_sum(xs3[0]**2)]
    input_dict_3 = {
        'ys': ys3,
        'xs': xs3,
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessian_symbolic_matrix_f64'
    }
    list_of_inputs.append(input_dict_3)

    # Input 4: Multiple tensors in xs
    g4 = tf.Graph()
    with g4.as_default():
        x4_1 = tf.constant(2.0, dtype=tf.float32)
        x4_2 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        xs4 = [x4_1, x4_2]
        ys4 = [x4_1**2 + tf.reduce_sum(x4_2**2)]
    input_dict_4 = {
        'ys': ys4,
        'xs': xs4,
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessian_multi_xs'
    }
    list_of_inputs.append(input_dict_4)

    # Input 5: Multiple tensors in ys (will be summed)
    g5 = tf.Graph()
    with g5.as_default():
        xs5 = [tf.constant([1.0, 2.0], dtype=tf.float32)]
        y5_1 = tf.reduce_sum(xs5[0]**2)
        y5_2 = tf.reduce_sum(xs5[0]**3)
        ys5 = [y5_1, y5_2]
    input_dict_5 = {
        'ys': ys5,
        'xs': xs5,
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessian_multi_ys'
    }
    list_of_inputs.append(input_dict_5)

    # Input 6: Higher-dimensional tensor (3D)
    g6 = tf.Graph()
    with g6.as_default():
        xs6 = [tf.constant(np.arange(8, dtype=np.float32).reshape((2, 2, 2)))]
        ys6 = [tf.reduce_sum(xs6[0]**4)]
    input_dict_6 = {
        'ys': ys6,
        'xs': xs6,
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessian_symbolic_3d'
    }
    list_of_inputs.append(input_dict_6)

    # Input 7: Using tf.Variable
    g7 = tf.Graph()
    with g7.as_default():
        xs7 = [tf.Variable([1.0, 2.0, 3.0], dtype=tf.float32)]
        ys7 = [tf.reduce_sum(xs7[0]**2)]
    input_dict_7 = {
        'ys': ys7,
        'xs': xs7,
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessian_variable'
    }
    list_of_inputs.append(input_dict_7)

    # Input 8: Empty tensor for xs
    g8 = tf.Graph()
    with g8.as_default():
        xs8 = [tf.constant(np.zeros((0,2)), dtype=tf.float32)]
        ys8 = [tf.reduce_sum(xs8[0]**2)]
    input_dict_8 = {
        'ys': ys8,
        'xs': xs8,
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessian_empty_input'
    }
    list_of_inputs.append(input_dict_8)

    # Input 9: More complex function
    g9 = tf.Graph()
    with g9.as_default():
        xs9 = [tf.constant([0.5, 1.5], dtype=tf.float64)]
        ys9 = [tf.exp(tf.reduce_sum(tf.math.sin(xs9[0])))]
    input_dict_9 = {
        'ys': ys9,
        'xs': xs9,
        'gate_gradients': True,
        'aggregation_method': None,
        'name': 'hessian_complex_func'
    }
    list_of_inputs.append(input_dict_9)

    # Input 10: Mixed precision in xs (requires casting for ops)
    g10 = tf.Graph()
    with g10.as_default():
        x10_1 = tf.constant(1.0, dtype=tf.float32)
        x10_2 = tf.constant([2.0, 3.0], dtype=tf.float64)
        xs10 = [x10_1, x10_2]
        ys10 = [tf.cast(x10_1, tf.float64)**2 + tf.reduce_sum(x10_2**2)]
    input_dict_10 = {
        'ys': ys10,
        'xs': xs10,
        'gate_gradients': False,
        'aggregation_method': None,
        'name': 'hessian_mixed_precision'
    }
    list_of_inputs.append(input_dict_10)

    return list_of_inputs

generated_inputs["tf.hessians"] = tf_hessians_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.hessians' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.hessians'.")

check_valid('tf.hessians', generated_inputs['tf.hessians'], lib="tf", suffix=0)

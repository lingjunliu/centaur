
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_data_experimental_from_variant_inputs():
    list_of_inputs = []
    
    variant1 = tf.constant([1, 2, 3])
    structure1 = [tf.TensorSpec(shape=(), dtype=np.int32)]
    list_of_inputs.append({"variant": variant1, "structure": structure1})
    
    variant2 = tf.constant([[1, 2], [3, 4]])
    structure2 = [tf.TensorSpec(shape=(2,), dtype=np.int32)]
    list_of_inputs.append({"variant": variant2, "structure": structure2})
    
    variant3 = tf.constant([1.0, 2.0, 3.0])
    structure3 = [tf.TensorSpec(shape=(), dtype=np.float32)]
    list_of_inputs.append({"variant": variant3, "structure": structure3})
    
    variant4 = tf.constant([[-1, 2], [3, -4]])
    structure4 = [tf.TensorSpec(shape=(2,), dtype=np.int32)]
    list_of_inputs.append({"variant": variant4, "structure": structure4})
    
    variant5 = tf.constant([True, False, True])
    structure5 = [tf.TensorSpec(shape=(), dtype=np.bool_)]
    list_of_inputs.append({"variant": variant5, "structure": structure5})

    variant6 = tf.constant([1, 2, 3, 4, 5])
    structure6 = [tf.TensorSpec(shape=(), dtype=np.int64)]
    list_of_inputs.append({"variant": variant6, "structure": structure6})

    variant7 = tf.constant([0.1, 0.2, 0.3])
    structure7 = [tf.TensorSpec(shape=(), dtype=np.float64)]
    list_of_inputs.append({"variant": variant7, "structure": structure7})
    
    return list_of_inputs

generated_inputs["tf.data.experimental.from_variant"] = tf_data_experimental_from_variant_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.from_variant' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.from_variant'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.from_variant', generated_inputs['tf.data.experimental.from_variant'], lib="tf", suffix=0)

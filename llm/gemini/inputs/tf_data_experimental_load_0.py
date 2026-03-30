
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_data_experimental_load_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    path = "path1"
    element_spec = []
    compression = "NONE"
    reader_func = None
    input_dict = {"path": path, "element_spec": element_spec, "compression": compression, "reader_func": reader_func}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With GZIP compression
    path = "path2"
    element_spec = []
    compression = "GZIP"
    reader_func = None
    input_dict = {"path": path, "element_spec": element_spec, "compression": compression, "reader_func": reader_func}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With element_spec (int64)
    path = "path3"
    element_spec = [tf.TensorSpec(shape=(), dtype=tf.int64, name=None)]
    compression = "NONE"
    reader_func = None
    input_dict = {"path": path, "element_spec": element_spec, "compression": compression, "reader_func": reader_func}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With element_spec (float32)
    path = "path4"
    element_spec = [tf.TensorSpec(shape=(None,), dtype=tf.float32, name=None)]
    compression = "NONE"
    reader_func = None
    input_dict = {"path": path, "element_spec": element_spec, "compression": compression, "reader_func": reader_func}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: With element_spec (string)
    path = "path5"
    element_spec = [tf.TensorSpec(shape=(), dtype=tf.string, name=None)]
    compression = "NONE"
    reader_func = None
    input_dict = {"path": path, "element_spec": element_spec, "compression": compression, "reader_func": reader_func}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: With element_spec (mixed types)
    path = "path6"
    element_spec = [tf.TensorSpec(shape=(), dtype=tf.int32, name=None), tf.TensorSpec(shape=(2,2), dtype=tf.float64, name=None)]
    compression = "NONE"
    reader_func = None
    input_dict = {"path": path, "element_spec": element_spec, "compression": compression, "reader_func": reader_func}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: reader_func is None
    path = "path7"
    element_spec = []
    compression = "NONE"
    reader_func = None
    input_dict = {"path": path, "element_spec": element_spec, "compression": compression, "reader_func": reader_func}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Different element spec shape
    path = "path8"
    element_spec = [tf.TensorSpec(shape=(1, 5, 10), dtype=tf.int32, name=None)]
    compression = "NONE"
    reader_func = None
    input_dict = {"path": path, "element_spec": element_spec, "compression": compression, "reader_func": reader_func}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multiple element specs
    path = "path9"
    element_spec = [tf.TensorSpec(shape=(), dtype=tf.int32, name=None), tf.TensorSpec(shape=(), dtype=tf.float32, name=None), tf.TensorSpec(shape=(), dtype=tf.string, name=None)]
    compression = "NONE"
    reader_func = None
    input_dict = {"path": path, "element_spec": element_spec, "compression": compression, "reader_func": reader_func}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: GZIP with Element Spec
    path = "path10"
    element_spec = [tf.TensorSpec(shape=(), dtype=tf.int64, name=None)]
    compression = "GZIP"
    reader_func = None
    input_dict = {"path": path, "element_spec": element_spec, "compression": compression, "reader_func": reader_func}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: With simple reader_func
    def custom_reader_func(datasets):
        return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

    path = "path11"
    element_spec = []
    compression = "NONE"
    reader_func = custom_reader_func

    input_dict = {"path": path, "element_spec": element_spec, "compression": compression, "reader_func": reader_func}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.load"] = tf_data_experimental_load_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.load' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.load'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.load', generated_inputs['tf.data.experimental.load'], lib="tf", suffix=0)

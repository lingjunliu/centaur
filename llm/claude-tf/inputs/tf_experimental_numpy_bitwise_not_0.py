
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

def tf_experimental_numpy_bitwise_not_inputs():
    list_of_inputs = []
    
    list_of_inputs.append({"x": np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)})
    list_of_inputs.append({"x": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)})
    list_of_inputs.append({"x": np.array([-1, -2, -3, -4], dtype=np.int32)})
    list_of_inputs.append({"x": np.array([1, -1, 0, 10, -10], dtype=np.int32)})
    list_of_inputs.append({"x": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)})
    list_of_inputs.append({"x": np.array(42, dtype=np.int32)})
    list_of_inputs.append({"x": np.array([0, 0, 0, 0], dtype=np.int32)})
    list_of_inputs.append({"x": np.array([1, 2, 3, 4, 5], dtype=np.int8)})
    list_of_inputs.append({"x": np.array([1000, 2000, 3000, -1000], dtype=np.int64)})
    list_of_inputs.append({"x": np.array([0, 1, 255, 128], dtype=np.uint8)})
    list_of_inputs.append({"x": np.array([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], dtype=np.int32)})
    list_of_inputs.append({"x": np.array([[100, 200, 300], [400, 500, 600]], dtype=np.int32)})
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.bitwise_not"] = tf_experimental_numpy_bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.bitwise_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.bitwise_not'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.bitwise_not', generated_inputs['tf.experimental.numpy.bitwise_not'], lib="tf", suffix=0)

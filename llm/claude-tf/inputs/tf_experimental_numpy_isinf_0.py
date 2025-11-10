
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

def tf_experimental_numpy_isinf_inputs():
    list_of_inputs = []
    
    list_of_inputs.append({"x": np.array([np.inf, 1.0, 2.0])})
    list_of_inputs.append({"x": np.array([-np.inf, 0.0, -1.0])})
    list_of_inputs.append({"x": np.array([np.inf, -np.inf, 0.0, 1.0])})
    list_of_inputs.append({"x": np.array([1.0, 2.0, 3.0, 4.0])})
    list_of_inputs.append({"x": np.array([[np.inf, 1.0], [-np.inf, 2.0]])})
    list_of_inputs.append({"x": np.array([[[np.inf, -np.inf], [1.0, 2.0]], [[3.0, 4.0], [5.0, np.inf]]])})
    list_of_inputs.append({"x": np.array(np.inf)})
    list_of_inputs.append({"x": np.array([np.nan, np.inf, -np.inf, 1.0])})
    list_of_inputs.append({"x": np.array([])})
    list_of_inputs.append({"x": np.array([1, 2, 3, 4], dtype=np.int32)})
    list_of_inputs.append({"x": np.array([[np.inf, 1.0, 2.0, -np.inf], [3.0, 4.0, 5.0, 6.0]])})
    list_of_inputs.append({"x": np.array([1e308, -1e308, 1e-308])})
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.isinf"] = tf_experimental_numpy_isinf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.isinf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.isinf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.isinf', generated_inputs['tf.experimental.numpy.isinf'], lib="tf", suffix=0)

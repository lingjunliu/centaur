
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

def tf_experimental_numpy_fix_inputs():
    list_of_inputs = []
    
    list_of_inputs.append({"x": np.array([1.2, 2.5, 3.7, 4.9])})
    list_of_inputs.append({"x": np.array([-1.2, -2.5, -3.7, -4.9])})
    list_of_inputs.append({"x": np.array([-2.3, 1.8, -0.5, 3.2])})
    list_of_inputs.append({"x": np.array([[1.5, -2.3], [3.7, -4.1]])})
    list_of_inputs.append({"x": np.array([[[1.2, -2.3], [3.4, -4.5]], [[5.6, -6.7], [7.8, -8.9]]])})
    list_of_inputs.append({"x": np.array(3.14159)})
    list_of_inputs.append({"x": np.array([0.0, -0.0, 0.5, -0.5])})
    list_of_inputs.append({"x": np.array([10.9, -10.9, 100.1, -100.1])})
    list_of_inputs.append({"x": np.array([1.0, 2.0, 3.0, -4.0, -5.0])})
    list_of_inputs.append({"x": np.array([0.1, -0.1, 0.9, -0.9])})
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.fix"] = tf_experimental_numpy_fix_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.fix' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.fix'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.fix', generated_inputs['tf.experimental.numpy.fix'], lib="tf", suffix=0)

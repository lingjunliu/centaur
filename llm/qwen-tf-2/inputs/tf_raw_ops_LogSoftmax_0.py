
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def generate_logsoftmax_inputs():
    inputs_list = []
    
    # Input 1: 2D tensor with float32 dtype
    logits1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict1 = {"name": "logsoftmax1", "logits": logits1}
    inputs_list.append(input_dict1)
    
    # Input 2: 2D tensor with float64 dtype
    logits2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict2 = {"name": "logsoftmax2", "logits": logits2}
    inputs_list.append(input_dict2)
    
    # Input 3: 2D tensor with half dtype
    logits3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16)
    input_dict3 = {"name": "logsoftmax3", "logits": logits3}
    inputs_list.append(input_dict3)
    
    # Input 4: 2D tensor with bfloat16 dtype
    logits4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_dict4 = {"name": "logsoftmax4", "logits": logits4}
    inputs_list.append(input_dict4)
    
    # Input 5: 2D tensor with negative values
    logits5 = np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict5 = {"name": "logsoftmax5", "logits": logits5}
    inputs_list.append(input_dict5)
    
    # Input 6: 2D tensor with mixed values including negative
    logits6 = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict6 = {"name": "logsoftmax6", "logits": logits6}
    inputs_list.append(input_dict6)
    
    # Input 7: 2D tensor with large values
    logits7 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    input_dict7 = {"name": "logsoftmax7", "logits": logits7}
    inputs_list.append(input_dict7)
    
    # Input 8: 2D tensor with zero values
    logits8 = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    input_dict8 = {"name": "logsoftmax8", "logits": logits8}
    inputs_list.append(input_dict8)
    
    # Input 9: 2D tensor with very small values
    logits9 = np.array([[0.001, 0.002], [0.003, 0.004]], dtype=np.float32)
    input_dict9 = {"name": "logsoftmax9", "logits": logits9}
    inputs_list.append(input_dict9)
    
    # Input 10: 2D tensor with float32 dtype and different values
    logits10 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict10 = {"name": "logsoftmax10", "logits": logits10}
    inputs_list.append(input_dict10)

    return inputs_list

generated_inputs["tf.raw_ops.LogSoftmax"] = generate_logsoftmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.LogSoftmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LogSoftmax'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.LogSoftmax', generated_inputs['tf.raw_ops.LogSoftmax'], lib="tf", suffix=0)

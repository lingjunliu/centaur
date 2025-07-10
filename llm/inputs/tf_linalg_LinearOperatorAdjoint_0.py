
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatoradjoint_inputs():
    list_of_inputs = []

    # Input 1
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    is_non_singular = True
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_1"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1. + 1j, 2.], [3., 4. - 1j]], dtype=np.complex64))
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_2"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "adjoint_op_3"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[5., 0.], [0., 2.]], dtype=np.float32))
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "adjoint_op_4"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32))
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_5"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "adjoint_op_6"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 2.], [2., 1.]], dtype=np.float32))
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_7"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[1., 0.], [0., -1.]], dtype=np.float32))
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_8"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    operator = tf.linalg.LinearOperatorFullMatrix(np.eye(4, dtype=np.float32))
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "adjoint_op_9"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    operator = tf.linalg.LinearOperatorFullMatrix(np.array([[0., 1.], [1., 0.]], dtype=np.float32))
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "adjoint_op_10"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint,
                  "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorAdjoint"] = tf_linalg_linearoperatoradjoint_inputs()

def check_valid(api, generated_inputs, lib="tf", suffix=0):
    import inspect
    import numpy as np

    def get_signature(api, lib="tf", suffix=0):
        if lib == "torch":
            func = eval("torch." + api)
        else:
            func = eval("tf." + api)
        sig = inspect.signature(func)
        return sig

    def get_type(domain, obj):
        if domain == "tensor":
            return "tensor"
        elif domain == "int":
            return "int"
        elif domain == "float":
            return "float"
        elif domain == "str":
            return "str"
        elif domain == "bool":
            return "bool"
        else:
            return "unknown"

    def get_ll(domain, value):
        if domain == "tensor":
            if isinstance(value, tf.linalg.LinearOperator):
                value_np = value.to_dense().numpy()
            else:
                value_np = value.numpy() if hasattr(value, 'numpy') else value
            
            if hasattr(value_np, 'size'):
                if hasattr(value_np, 'shape') and len(value_np.shape) == 0:
                    range_val = [value_np.item(), value_np.item()]
                elif value_np.size > 0:
                    range_val = [np.min(value_np), np.max(value_np)]
                else:
                    range_val = [0, 0]
            else:
                range_val = [0, 0]

        elif domain == "int":
            range_val = [value, value]
        elif domain == "float":
            range_val = [value, value]
        elif domain == "str":
            range_val = [value, value]
        elif domain == "bool":
            range_val = [value, value]
        else:
            range_val = [0, 0]
        return range_val
        
    def get_abstract_input(concrete, signature):
        abstract = {}
        for arg, param in signature.parameters.items():
            if arg in concrete:
                domain = signature.parameters[arg].annotation
                abstract[arg] = get_ll(domain, concrete[arg])
        return abstract

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorAdjoint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorAdjoint'.")

check_valid('tf.linalg.LinearOperatorAdjoint', generated_inputs['tf.linalg.LinearOperatorAdjoint'], lib="tf", suffix=0)

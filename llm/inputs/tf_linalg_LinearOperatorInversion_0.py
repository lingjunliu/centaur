
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorinversion_inputs():
    list_of_inputs = []

    # Input 1
    matrix1 = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix1)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op1"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    matrix2 = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix2)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op2"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    matrix3 = np.array([[4.0, 0.0], [0.0, 9.0]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix3)
    is_non_singular = False
    is_self_adjoint = False
    is_positive_definite = False
    is_square = True
    name = "inv_op3"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    matrix4 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix4)
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = None
    name = "inv_op4"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    matrix5 = np.array([[5.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix5)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op5"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    matrix6 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix6)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = ""
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    matrix7 = np.eye(3, dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix7)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op7"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    matrix8 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix8)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op8"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    matrix9 = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix9)
    is_non_singular = False
    is_self_adjoint = True
    is_positive_definite = True
    is_square = True
    name = "inv_op9"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    matrix10 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    operator = tf.linalg.LinearOperatorFullMatrix(matrix10)
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "inv_op10"
    input_dict = {"operator": operator, "is_non_singular": is_non_singular, "is_self_adjoint": is_self_adjoint, "is_positive_definite": is_positive_definite, "is_square": is_square, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.LinearOperatorInversion"] = tf_linalg_linearoperatorinversion_inputs()

def check_valid(api, input_list, lib="tf", suffix=0):
  import inspect

  def get_signature(api, lib="tf", suffix=0):
    if lib == "torch":
      s = inspect.signature(eval(api))
    elif lib == "tf":
      s = inspect.signature(eval(api))
    elif lib == "jax":
      s = inspect.signature(eval(api))
    
    input_names = list(s.parameters)
    input_types = []
    for input_name in input_names:
        param = s.parameters[input_name]
        if param.annotation != inspect.Parameter.empty:
            input_types.append(param.annotation)
        else:
            input_types.append(None)
            
    return dict(zip(input_names, input_types))

  def get_abstract_input(input_dict, signature):
    abstract = {}
    for arg in input_dict:
      value = input_dict[arg]
      domain = signature[arg]

      if str(domain).startswith("<class 'tensorflow.python.ops.linalg.linalg_impl.LinearOperatorFullMatrix'>"):
          range_val = [np.min(value.to_dense().numpy()), np.max(value.to_dense().numpy())] if value.to_dense().numpy().size > 0 else [0, 0]
      elif str(domain) == "<class 'bool'>":
        range_val = [True, False]
      elif str(domain) == "<class 'str'>":
        range_val = ["", "test"]
      else:
        raise Exception(f"Type {domain} is unsupported")

      abstract[arg] = get_ll(domain, input_dict[arg])
    return abstract
  
  def get_ll(domain, value):
    import numpy as np
    if str(domain).startswith("<class 'tensorflow.python.ops.linalg.linalg_impl.LinearOperatorFullMatrix'>"):
        range_val = [np.min(value.to_dense().numpy()), np.max(value.to_dense().numpy())] if value.to_dense().numpy().size > 0 else [0, 0]
    elif str(domain) == "<class 'bool'>":
      range_val = [True, False]
    elif str(domain) == "<class 'str'>":
      range_val = ["", "test"]
    else:
      raise Exception(f"Type {domain} is unsupported")
    return range_val

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.LinearOperatorInversion' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorInversion'.")

check_valid('tf.linalg.LinearOperatorInversion', generated_inputs['tf.linalg.LinearOperatorInversion'], lib="tf", suffix=0)

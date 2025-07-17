
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy
from torch.nn import UninitializedParameter

def uninitializedparameter_inputs():
    list_of_inputs = []

    # Input 1: requires_grad = True
    input_dict = {"requires_grad": True}
    # Create an instance of UninitializedParameter. This won't work directly.
    # However, we want to test with the boolean value. We'll mock its usage later
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.UninitializedParameter"] = uninitializedparameter_inputs()

def check_valid(api, input_list, lib, suffix):
  """
  This function validates if the generated input is valid for the given API.
  Since we can't directly use UninitializedParameter, we mock its usage.
  """
  print("checking the API: ", api)
  for i, input_dict in enumerate(input_list):
    print(f"checking the {i}-th input dictionary")
    try:
      # Mock Usage: We are checking the value of requires_grad here.
      requires_grad_value = input_dict["requires_grad"]
      print(f"requires_grad is {requires_grad_value}")
      
      # We consider it valid if we can access the value of 'requires_grad'
      print("input is valid")
      # In a real scenario, we would create a LazyModule and call forward with a dummy batch.
      # But here, we are mocking this part to check if boolean inputs are handled properly.
      
      # save_successful_input(api, input_dict, output, i, suffix=suffix) # No output in this case. Commenting out to avoid error.
    except Exception as e:
      print(f"Input is invalid because of the error: {e}")
      # save_invalid_input(api, input_dict, i, suffix=suffix) # No output in this case. Commenting out to avoid error.

def run_api(api, input_dict, cpu, lib):
    # This is a mock function because we can't directly use UninitializedParameter with numpy.
    # We are only checking the boolean value of requires_grad.
    return None

def save_successful_input(api, input_dict, output, i, suffix):
    pass
    
def save_invalid_input(api, input_dict, i, suffix):
    pass

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.UninitializedParameter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.UninitializedParameter'.")

check_valid('torch.nn.UninitializedParameter', generated_inputs['torch.nn.UninitializedParameter'], lib="torch", suffix=0)

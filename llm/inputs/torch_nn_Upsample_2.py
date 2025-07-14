
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def upsample_inputs():
    list_of_inputs = []

    def create_input_dict(size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, input=None):
      input_dict = {
          "size": size,
          "scale_factor": scale_factor,
          "mode": mode,
          "align_corners": align_corners,
          "recompute_scale_factor": recompute_scale_factor,
          "input": input
      }
      
      if mode not in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        input_dict["align_corners"] = None
      elif align_corners is None:
        input_dict["align_corners"] = False  

      if scale_factor is None:
        input_dict["recompute_scale_factor"] = None
      elif recompute_scale_factor is None:
          input_dict["recompute_scale_factor"] = False
      
      return input_dict

    # Input 1
    input1 = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    input_dict1 = create_input_dict(scale_factor=2.0, mode="nearest", input=input1)
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    input_dict2 = create_input_dict(scale_factor=2.0, mode="bilinear", align_corners=False, input=input2)
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    input_dict3 = create_input_dict(scale_factor=2.0, mode="bilinear", align_corners=True, input=input3)
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.zeros((3, 3), dtype=np.float32).reshape(1, 1, 3, 3)
    input4[:, :, :2, :2] = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    input_dict4 = create_input_dict(scale_factor=2.0, mode="bilinear", align_corners=False, input=input4)
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.zeros((3, 3), dtype=np.float32).reshape(1, 1, 3, 3)
    input5[:, :, :2, :2] = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    input_dict5 = create_input_dict(scale_factor=2.0, mode="bilinear", align_corners=True, input=input5)
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6
    input6 = np.arange(1, 28, dtype=np.float32).reshape(1, 1, 3, 3, 3)
    input_dict6 = create_input_dict(scale_factor=2.0, mode="trilinear", align_corners=False, input=input6)
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7
    input7 = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
    input_dict7 = create_input_dict(size=(6,6), mode="bicubic", align_corners=True, input=input7)
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8
    input8 = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    input_dict8 = create_input_dict(size=(4,4), mode="nearest", input=input8)
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.arange(1, 9, dtype=np.float32).reshape(1, 1, 2, 2, 2)
    input_dict9 = create_input_dict(scale_factor=1.5, mode="trilinear", align_corners=False, input=input9)
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10
    input10 = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    input_dict10 = create_input_dict(scale_factor=2.0, mode="bilinear", align_corners=True, recompute_scale_factor=True, input=input10)
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11
    input11 = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    input_dict11 = create_input_dict(scale_factor=2.5, mode="bilinear", align_corners=False, recompute_scale_factor=True, input=input11)
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # Input 12
    input12 = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    input_dict12 = create_input_dict(size=(4,4), mode="bilinear", align_corners=True, input=input12)
    list_of_inputs.append(copy.deepcopy(input_dict12))
    
    # Input 13
    input13 = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    input_dict13 = create_input_dict(size=(4,4), mode="bilinear", align_corners=False, input=input13)
    list_of_inputs.append(copy.deepcopy(input_dict13))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Upsample_2"] = upsample_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Upsample_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Upsample_2'.")

check_valid('torch.nn.Upsample', generated_inputs['torch.nn.Upsample_2'], lib="torch", suffix=2)

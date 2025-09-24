
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def nansum_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([1.0, float('nan'), -3.5, 2.5], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (0,),
        "keepdim": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.tensor([[1.0, 2.0], [float('nan'), 4.0]], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (0,),
        "keepdim": True,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.tensor([[1, -2, 3], [4, 5, -6]], dtype=torch.int64).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (1,),
        "keepdim": False,
        "dtype": torch.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]],
                              [[-1.0, 2.0], [float('nan'), 5.0]]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (1, 2),
        "keepdim": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.tensor([[[1.0, 2.0, float('nan'), -4.0],
                               [5.0, -6.0, 7.0, float('nan')],
                               [9.0, 10.0, -11.0, 12.0]],
                              [[float('nan'), -2.5, 3.0, 4.0],
                               [5.5, float('nan'), -7.5, 8.0],
                               [9.5, -10.0, 11.0, float('nan')]]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (-1,),
        "keepdim": True,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.tensor([[[[1.0, float('nan'), -3.0],
                                [4.0, 5.0, float('nan')]],
                               [[-1.0, 2.0, 3.0],
                                [float('nan'), -5.0, 6.0]]],
                              [[[7.0, 8.0, float('nan')],
                                [-9.0, 10.0, 11.0]],
                               [[float('nan'), -12.0, 13.0],
                                [14.0, 15.0, float('nan')]]],
                              [[[1.5, -2.5, 3.5],
                                [float('nan'), -4.5, 5.5]],
                               [[6.5, float('nan'), -7.5],
                                [8.5, 9.5, 10.5]]],
                              [[[float('nan'), 0.0, -1.0],
                                [2.0, 3.0, 4.0]],
                               [[-5.0, float('nan'), 6.0],
                                [7.0, 8.0, 9.0]]]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (0, 3),
        "keepdim": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.tensor([[1.0, -2.0, float('nan')],
                              [4.5, float('nan'), -6.5],
                              [7.0, 8.0, 9.0],
                              [float('nan'), -10.0, 11.0],
                              [12.0, -13.0, 14.0]], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (0, 1),
        "keepdim": True,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.tensor([[[-1, 2],
                               [3, -4],
                               [5, -6],
                               [7, 8]],
                              [[-9, 10],
                               [11, -12],
                               [13, -14],
                               [15, 16]],
                              [[-17, 18],
                               [19, -20],
                               [21, -22],
                               [23, 24]]], dtype=torch.int16).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (0,),
        "keepdim": False,
        "dtype": torch.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.tensor([[[[1.0, float('nan')], [3.0, 4.0]],
                               [[-5.0, 6.0], [float('nan'), 8.0]]],
                              [[[9.0, -10.0], [11.0, float('nan')]],
                               [[13.0, 14.0], [-15.0, 16.0]]]], dtype=torch.float16).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (1, 2),
        "keepdim": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.tensor([[True, False, True],
                              [False, True, True],
                              [True, True, False]], dtype=torch.bool).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (1,),
        "keepdim": True,
        "dtype": torch.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.tensor(np.array([[[[1.0, float('nan'), 3.0, -4.0, 5.0],
                                         [6.0, -7.0, float('nan'), 9.0, -10.0],
                                         [11.0, 12.0, -13.0, 14.0, float('nan')],
                                         [16.0, -17.0, 18.0, 19.0, -20.0]],
                                        [[-1.5, 2.5, float('nan'), -4.5, 5.5],
                                         [6.5, float('nan'), 8.5, -9.5, 10.5],
                                         [11.5, -12.5, 13.5, float('nan'), 15.5],
                                         [16.5, 17.5, -18.5, 19.5, 20.5]],
                                        [[float('nan'), -2.0, 3.0, 4.0, -5.0],
                                         [6.0, 7.0, float('nan'), -9.0, 10.0],
                                         [11.0, -12.0, 13.0, 14.0, float('nan')],
                                         [16.0, 17.0, -18.0, 19.0, 20.0]]],
                                      [[[-1.0, 2.0, -3.0, 4.0, float('nan')],
                                        [6.0, -7.0, 8.0, float('nan'), 10.0],
                                        [11.0, float('nan'), 13.0, -14.0, 15.0],
                                        [16.0, -17.0, 18.0, 19.0, -20.0]],
                                       [[1.25, -2.25, 3.25, float('nan'), -5.25],
                                        [6.25, 7.25, -8.25, 9.25, float('nan')],
                                        [11.25, -12.25, 13.25, 14.25, -15.25],
                                        [float('nan'), 17.25, 18.25, -19.25, 20.25]],
                                       [[-1.75, float('nan'), 3.75, -4.75, 5.75],
                                        [6.75, 7.75, -8.75, 9.75, 10.75],
                                        [float('nan'), -12.75, 13.75, 14.75, -15.75],
                                        [16.75, 17.75, float('nan'), -19.75, 20.75]]]], dtype=np.float32)).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (-4, -2),
        "keepdim": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.tensor([float('nan')], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "dim": (0,),
        "keepdim": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nansum_3"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nansum_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_3'.")


check_valid('torch.nansum', generated_inputs['torch.nansum_3'], lib="torch", suffix=3)

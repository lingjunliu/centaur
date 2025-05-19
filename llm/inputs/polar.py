
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def polar_inputs():
    list_of_inputs = []

    abs_val = np.array([1.0, 2.0, 3.0])
    angle_val = np.array([0.0, np.pi/2, np.pi])
    input_dict = {"abs": abs_val, "angle": angle_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    abs_val = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    angle_val = np.array([[np.pi, 0.0], [np.pi/4, np.pi/2]])
    input_dict = {"abs": abs_val, "angle": angle_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    abs_val = np.array([1, 2, 3], dtype=np.int32)
    angle_val = np.array([0, 1, 2], dtype=np.int32) * np.pi / 4
    input_dict = {"abs": abs_val.astype(np.float32), "angle": angle_val.astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    abs_val = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    angle_val = np.array([[[0.0, np.pi], [np.pi/2, np.pi/4]], [[np.pi/3, np.pi/6], [np.pi/8, np.pi/5]]])
    input_dict = {"abs": abs_val, "angle": angle_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    abs_val = np.array([1.5, 2.5, 3.5])
    angle_val = np.array([-np.pi/2, -np.pi/4, 0.0])
    input_dict = {"abs": abs_val, "angle": angle_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    abs_val = np.array([0.0, 0.0, 0.0])
    angle_val = np.array([0.0, np.pi, -np.pi])
    input_dict = {"abs": abs_val, "angle": angle_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    abs_val = np.array([1.0])
    angle_val = np.array([np.pi/2])
    input_dict = {"abs": abs_val, "angle": angle_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = polar_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('polar', generated_inputs)

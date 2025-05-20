
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def set_anomaly_enabled_inputs():
    list_of_inputs = []
    
    input1 = {'mode': True}
    list_of_inputs.append(copy.deepcopy(input1))
    
    input2 = {'mode': False}
    list_of_inputs.append(copy.deepcopy(input2))
    
    input3 = {'mode': np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input3))
    
    input4 = {'mode': np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input4))
    
    input5 = {'mode': np.array(True)}
    list_of_inputs.append(copy.deepcopy(input5))

    input6 = {'mode': np.array(False)}
    list_of_inputs.append(copy.deepcopy(input6))
    
    return list_of_inputs

generated_inputs = set_anomaly_enabled_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_anomaly_enabled', generated_inputs)

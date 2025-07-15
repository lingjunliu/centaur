import numpy as np
from learner.invariant_inference import get_invariants
from utils.defaults import *
from utils.new_api_utils import get_signature, get_lib_version

############### get definitions per api ################

'''
    Get definition per API with an empty random candidate
'''
def get_definition(api, z3=False, lib="torch", suffix=0, use_reference=False):
    signature = get_signature(api, lib=lib, suffix=suffix)
    domain_limits = domain_limits_torch if lib == "torch" else domain_limits_tf
    
    definition = {
        "api": api,
        "signature": signature,
        "ruleset": get_invariants(api, suffix=suffix, lib=lib, use_reference=use_reference),
        "random_candidate": {},
        "arg_order": list(signature.keys()),
        "limits": [],
        "suffix": suffix
    }
    
    definition["random_candidate"] = {}
    
    for arg, domain in signature.items():
        if domain == "tensor_list":
            domain = "tensor"   # hack until tensor_list is supported
        
        # Limits
        definition["limits"].append(domain_limits[domain])
        definition["limits"].append(domain_limits[f'{domain}_dtype'])
        definition["limits"].append(domain_limits[f'{domain}_value_range'])
    
    return definition

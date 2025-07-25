import sys, pickle, torch, os
from utils.misc import create_subdir, get_tmp_dir
from utils.api_utils import get_arglist
from utils.new_api_utils import get_signature, get_lib_version, get_n_variations
import logging
from utils.defaults import *

logger = logging.getLogger(__name__)

def print_arg(arg, prefix):
    if isinstance(arg, torch.Tensor):        
        try:
            logger.info(f"{prefix} shape: {arg.shape}, dtype: {arg.dtype}, range: [{torch.min(arg)},{torch.max(arg)}]")
        except:
            logger.info(f"{prefix} shape: {arg.shape}, dtype: {arg.dtype}")
    else:
        logger.info(f"{prefix} value: {arg}, dtype: {type(arg)}")

def match_types(arg, domain):
    if domain == "tensor" or domain == "tensor_list":
        return isinstance(arg, torch.Tensor)
    elif domain == "integer":
        return isinstance(arg, int)
    elif domain == "float":
        return isinstance(arg, float)
    elif domain == "dtype":
        return isinstance(arg, torch.dtype)
    elif domain == "string":
        return isinstance(arg, str)
    elif domain == "tuple":
        return isinstance(arg, tuple)
    elif domain == "list":
        return isinstance(arg, list)
    elif domain == "boolean":
        return isinstance(arg, bool)
    else:
        return False
        
def match_values(arg, domain, lib="torch"):
    invalid_dtype = 0
    invalid_length = 0
    invalid_value = 0
    invalid_ndim = 0
    invalid_dimsize = 0
    invalid_range = 0
    list_of_string_values = list_of_string_values_torch if lib == "torch" else list_of_string_values_tf
    domain_limits = domain_limits_torch if lib == "torch" else domain_limits_tf
    if domain == "dtype":
        if match_types(arg, domain):
            if not arg in list_of_available_dtypes:
                invalid_value += 1
        else:
            invalid_dtype += 1
    elif domain == "string":
        if match_types(arg, domain):
            if not arg in list_of_string_values:
                invalid_value += 1
        else:
            invalid_dtype += 1
    elif domain in ["list", "tuple"]:
        if match_types(arg, domain):
            if len(arg) < domain_limits[domain][2] or len(arg) > domain_limits[domain][3]:
                invalid_length += 1
            for v in arg:
                if v < domain_limits[domain][0] or v > domain_limits[domain][1]:
                    invalid_value += 1
        else:
            invalid_dtype += 1
    elif domain in ["float", "boolean", "integer"]:
        if match_types(arg, domain):
            if arg < domain_limits[domain][0] or arg > domain_limits[domain][1]:
                invalid_value += 1
        else:
            invalid_dtype += 1
    elif domain in ["tensor", "tensor_list"]:
        if match_types(arg, domain):
            shape = arg.shape
            if len(shape) < domain_limits["tensor"][2] or len(shape) > domain_limits["tensor"][3]:
                invalid_ndim += 1
            for v in shape:
                if v < domain_limits["tensor"][0] or v > domain_limits["tensor"][1]:
                    invalid_dimsize += 1
            try:
                min_val, max_val = torch.min(arg), torch.max(arg)
                if min_val < domain_limits["tensor_value_range"][0] or max_val > domain_limits["tensor_value_range"][1]:
                    invalid_range += 1
            except:
                pass
        else:
            invalid_dtype += 1
    
    return invalid_dtype, invalid_length, invalid_value, invalid_ndim, invalid_dimsize, invalid_range

def main():
    # Debug params
    multiply = "branch"  # set this to "branch" to multiply by the new_branches, "count" to count the instances only

    api = sys.argv[1]
    lib = sys.argv[2] if len(sys.argv) > 2 else "torch"

    api = get_lib_version(api, lib=lib)  # Get the API version

    output_dir = create_subdir(get_tmp_dir(), "debug_coverage")
    filename_new_br = os.path.join(output_dir, f"{api}.csv")

    if not os.path.exists(filename_new_br):
        print(f"{filename_new_br} does not exist")
        return
    
    logfile = f'{output_dir}/{api}_abstracts.log'
    csvfile = f'{output_dir}/{api}_stats.csv'
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,                                     # Minimum log level
        format='%(message)s',                                   # Log format
        filename=logfile,                                       # Log file path
        filemode="w"                                            # Append/Write mode
    )

    with open(filename_new_br, "r") as f:
        list_of_files = [line.strip().split(',') for line in f.readlines()]
    
    n_variants = get_n_variations(api, lib="torch")  # Get the number of variations for the API

    if n_variants > 1:
        api_signatures = [get_signature(api, lib="torch", suffix=i) for i in range(1, n_variants+1)]  # Get signatures for all variations
    else:
        api_signatures = [get_signature(api, lib="torch", suffix=0)]  # Get the API signature for the given API
    
    # Categories
    missing_params = 0
    different_dtype = 0
    length_mismatches = 0
    value_mismatches = 0
    ndim_mismatches = 0
    dimsize_mismatches = 0
    range_mismatches = 0
    
    for file, new_branches in list_of_files:
        with open(file, "rb") as f:
            pkl_dict = pickle.load(f)
            for torch_api, input_dict in pkl_dict.items():
                n_args = len(input_dict['args']) + len(input_dict['kwargs'].keys())
                arg_list = get_arglist(torch_api, n_args)
                logger.info(f"Args: {arg_list}")

                logger.info(f"New Branches: {new_branches} | API: {torch_api}")
                logger.info("args:")
                count = 0
                mult = int(new_branches) if multiply.lower() == "branch" else 1
                for arg in input_dict['args']:
                    name = arg_list[count]
                    count += 1
                    print_arg(arg, f"{count} | name: {name},")

                    api_signature = api_signatures[0]  # Default to the first signature
                    param_missing = True
                    for signature in api_signatures:
                        if name not in signature:
                            continue
                        else:
                            api_signature = signature
                            param_missing = False
                            break

                    if param_missing:
                        missing_params += mult
                    else:
                        # stat
                        domain = api_signature[name]
                        invalid_dtype, invalid_length, invalid_value, invalid_ndim, invalid_dimsize, invalid_range = match_values(arg, domain)
                        length_mismatches += invalid_length*mult
                        value_mismatches += invalid_value*mult
                        ndim_mismatches += invalid_ndim*mult
                        dimsize_mismatches += invalid_dimsize*mult
                        range_mismatches += invalid_range*mult
                        different_dtype += invalid_dtype*mult
                
                if len(input_dict['kwargs']) > 0:
                    logger.info("kwargs:")
                for name, kwarg in input_dict['kwargs'].items():
                    count += 1
                    print_arg(kwarg, f"{count} | name: {name},")
                    
                    api_signature = api_signatures[0]  # Default to the first signature
                    param_missing = True
                    for signature in api_signatures:
                        if name not in signature:
                            continue
                        else:
                            api_signature = signature
                            param_missing = False
                            break

                    if param_missing:
                        missing_params += mult
                    else:
                        # stat
                        domain = api_signature[name]
                        invalid_dtype, invalid_length, invalid_value, invalid_ndim, invalid_dimsize, invalid_range = match_values(arg, domain)
                        length_mismatches += invalid_length*mult
                        value_mismatches += invalid_value*mult
                        ndim_mismatches += invalid_ndim*mult
                        dimsize_mismatches += invalid_dimsize*mult
                        range_mismatches += invalid_range*mult
                        different_dtype += invalid_dtype*mult
                
                logger.info("")

    logger.info("-----"*5)
    logger.info(f"missing_params: {missing_params}")
    logger.info(f"different_dtype: {different_dtype}")
    logger.info(f"length_mismatches: {length_mismatches}")
    logger.info(f"value_mismatches: {value_mismatches}")
    logger.info(f"ndim_mismatches: {ndim_mismatches}")
    logger.info(f"dimsize_mismatches: {dimsize_mismatches}")
    logger.info(f"range_mismatches: {range_mismatches}")

    with open(csvfile, "w") as f:
        f.write(f"{api},{missing_params},{different_dtype},{length_mismatches},{value_mismatches},{ndim_mismatches},{dimsize_mismatches},{range_mismatches}\n")

    print(f"Logged abstract info to {logfile}")
    print(f"Wrote stats to {csvfile}")
if __name__ == "__main__":
    main()
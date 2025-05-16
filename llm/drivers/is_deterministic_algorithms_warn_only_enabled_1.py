import numpy as np
import os

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if not cpu:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if 'warn_only' in input_dict:
        warn_only = input_dict['warn_only']
    else:
        warn_only = False
    
    if not cpu:
      torch.cuda.init()

    torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    result = torch.is_deterministic_algorithms_warn_only_enabled()

    return {'result': np.array(result, dtype=bool)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if 'warn_only' in input_dict:
        warn_only = input_dict['warn_only']
    else:
        warn_only = False
    
    os.environ['TF_DETERMINISTIC_OPS'] = '1' if warn_only else '0'
    tf.config.experimental.enable_op_determinism()
    
    result = os.environ.get('TF_DETERMINISTIC_OPS') == '1'
    return {'result': np.array(result, dtype=bool)}

def main():
    A_TOL = 0.01
    input_data = {
        'warn_only': True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        'warn_only': False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
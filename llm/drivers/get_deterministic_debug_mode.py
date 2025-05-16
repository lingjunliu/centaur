import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if not cpu:
        torch.cuda.init()

    if torch.cuda.is_available() and not cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    result = torch.get_deterministic_debug_mode()

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    result = tf.config.experimental.get_visible_devices('GPU')
    if len(result) > 0 and not cpu:
        result = True
    else:
        result = False
    
    return {"result": np.array(result)}

def main():
    A_TOL = 0.01
    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
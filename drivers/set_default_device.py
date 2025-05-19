import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    device = 'cpu' if cpu else 'cuda'
    torch.set_default_device(device)
    
    if not cpu:
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('cpu')

    return {"result": np.array(0)}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device = '/CPU:0'
    else:
        device = '/GPU:0'

    with tf.device(device):
        pass

    return {"result": np.array(0)}


def main():
    A_TOL = 0.01
    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
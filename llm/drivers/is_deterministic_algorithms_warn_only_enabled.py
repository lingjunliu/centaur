import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    input = torch.tensor(input_dict.get("input"))

    if not cpu:
        input = input.cuda()

    result = torch.is_deterministic_algorithms_warn_only_enabled()

    if not cpu:
        pass

    return {'result': np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    input = input_dict.get("input")
    try:
        tf.config.experimental.enable_op_determinism()
        result = True
    except RuntimeError:
        result = False

    return {'result': np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1]),
    }

    torch.use_deterministic_algorithms(True)
    torch_result = torch_version(input_data)

    try:
        tf.config.experimental.enable_op_determinism()
        tf_result = tensorflow_version(input_data)
    except RuntimeError:
        tf.config.experimental.enable_op_determinism()
        tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
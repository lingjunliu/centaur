import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if not hasattr(torch, "clear_autocast_cache"):
        return {"result": None}

    if not cpu and torch.cuda.is_available():
        torch.cuda.init()

    torch.clear_autocast_cache()

    return {"result": None}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    tf.config.optimizer.set_jit(False)

    return {"result": None}


def main():
    A_TOL = 0.01

    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"]

    print("Success")


if __name__ == "__main__":
    main()
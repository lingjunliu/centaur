import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    try:
        raise torch.jit.Error(input_dict["msg"])
    except torch.jit.Error as e:
        result = str(e)

    return {"result": result}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    try:
        raise Exception(input_dict["msg"])
    except Exception as e:
        result = str(e)

    return {"result": result}


def main():
    A_TOL = 0.01
    input_data = {
        "msg": "This is a test error message."
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()
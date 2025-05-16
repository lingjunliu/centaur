import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
from torch.nn.functional import scaled_dot_product_attention

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.attention import can_use_flash_attention

    grad_enabled = input_dict["grad_enabled"]
    query = torch.tensor(input_dict["query"])
    key = torch.tensor(input_dict["key"])
    value = torch.tensor(input_dict["value"])
    mask = input_dict.get("mask", None)
    if mask is not None:
        mask = torch.tensor(input_dict["mask"])
        if not cpu:
            mask = mask.cuda()
    else:
        mask = None

    if not cpu:
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()

    with torch.set_grad_enabled(grad_enabled):
        try:
            result = can_use_flash_attention(query, key, value)
        except Exception as e:
            result = False

    if not cpu:
        result = result

    return {"result": np.array(result)}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    grad_enabled = input_dict["grad_enabled"]
    query = input_dict["query"]
    key = input_dict["key"]
    value = input_dict["value"]
    mask = input_dict.get("mask", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        result = False
        return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "grad_enabled": False,
        "query": np.random.rand(2, 3, 4, 8).astype(np.float32),
        "key": np.random.rand(2, 5, 4, 8).astype(np.float32),
        "value": np.random.rand(2, 5, 4, 8).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    input_data = {
        "grad_enabled": True,
        "query": np.random.rand(2, 3, 4, 8).astype(np.float32),
        "key": np.random.rand(2, 5, 4, 8).astype(np.float32),
        "value": np.random.rand(2, 5, 4, 8).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    input_data = {
        "grad_enabled": False,
        "query": np.random.rand(2, 3, 4, 9).astype(np.float32),
        "key": np.random.rand(2, 5, 4, 9).astype(np.float32),
        "value": np.random.rand(2, 5, 4, 9).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()
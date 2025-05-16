import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    indices = torch.tensor(input_dict.get("indices", [0])).long()
    dim = input_dict.get("dim", 0)
    keepdim = input_dict.get("keepdim", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        indices = indices.cuda()

    result = torch.index_select(input_tensor, dim, indices)
    if not keepdim:
        result = torch.squeeze(result)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        indices = tf.constant(input_dict.get("indices", [0]))
        dim = input_dict.get("dim", 0)
        keepdim = input_dict.get("keepdim", False)

        result = tf.gather(input_tensor, indices, axis=dim)
        if not keepdim:
            result = tf.squeeze(result)

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        "indices": np.array([0, 2], dtype=np.int32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        "indices": np.array([0, 1], dtype=np.int32),
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        "indices": np.array([0, 2], dtype=np.int32),
        "dim": 0,
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()
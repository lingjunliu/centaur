import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["data"])
    requires_grad = input_dict.get("requires_grad", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    param = torch.nn.Parameter(input_tensor, requires_grad=requires_grad)

    if not cpu:
        param = param.cpu()

    return {"result": param.data.numpy(), "requires_grad": param.requires_grad}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    data = input_dict["data"]
    requires_grad = input_dict.get("requires_grad", False)

    if cpu:
        with tf.device("/cpu:0"):
            variable = tf.Variable(data, trainable=requires_grad)
            result = variable.numpy()
    else:
        with tf.device("/gpu:0"):
            variable = tf.Variable(data, trainable=requires_grad)
            result = variable.numpy()

    return {"result": result, "requires_grad": requires_grad}

def main():
    A_TOL = 0.01

    input_data = {
        "data": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "requires_grad": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert torch_result["requires_grad"] == tf_result["requires_grad"], "Requires grad do not match"

    input_data = {
        "data": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "requires_grad": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert torch_result["requires_grad"] == tf_result["requires_grad"], "Requires grad do not match"

    input_data = {
        "data": np.array([7.0, 8.0, 9.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert torch_result["requires_grad"] == tf_result["requires_grad"], "Requires grad do not match"

    print("Success")

if __name__ == "__main__":
    main()
import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):

    input_tensor = torch.tensor(input_dict["input"])
    alpha = input_dict.get("alpha", 1.6732632423543772848170429916717)
    scale = input_dict.get("scale", 1.0507009873554804934193349852946)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = scale * torch.where(input_tensor >= 0, input_tensor, alpha * (torch.exp(input_tensor) - 1))

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        alpha = input_dict.get("alpha", 1.6732632423543772848170429916717)
        scale = input_dict.get("scale", 1.0507009873554804934193349852946)

        result = scale * tf.where(input_tensor >= 0.0, input_tensor, alpha * (tf.exp(input_tensor) - 1))

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    lambd = input_dict.get("lambd", 0.5)

    if not cpu:
        input_tensor = input_tensor.cuda()

    softshrink = torch.nn.Softshrink(lambd=lambd)
    result = softshrink(input_tensor)

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
        lambd = input_dict.get("lambd", 0.5)

        result = tf.where(input_tensor > lambd, input_tensor - lambd,
                         tf.where(input_tensor < -lambd, input_tensor + lambd,
                                  tf.zeros_like(input_tensor)))
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-2, -0.5, 0, 0.5, 2], dtype=np.float32),
        "lambd": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
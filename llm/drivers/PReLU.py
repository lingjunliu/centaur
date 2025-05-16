import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    scale = 1.0
    zero_point = 0
    dtype = torch.quint8

    q_input = torch.quantize_per_tensor(input_tensor, scale=scale, zero_point=zero_point, dtype=dtype)

    alpha_value = input_dict.get("init", 0.25)

    result = torch.nn.quantized.modules.PReLU(scale, zero_point, weight=torch.tensor([alpha_value]))
    if not cpu:
      result = result.cuda()
    result = result(q_input)

    if not cpu:
        result = result.cpu()

    result = torch.dequantize(result)
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        init_value = input_dict.get("init", 0.25)
        alpha = tf.Variable(initial_value=init_value, dtype=tf.float32)

        result = tf.maximum(input_tensor, 0.0) + alpha * tf.minimum(input_tensor, 0.0)


        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        "init": 0.25
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
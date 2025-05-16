import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.quantized as nnq

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    alpha = input_dict.get("alpha", 1.0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    scale = 1.0
    zero_point = 0
    dtype = torch.quint8

    q_input = torch.quantize_per_tensor(input_tensor, scale=scale, zero_point=zero_point, dtype=dtype)

    elu = nnq.ELU(alpha=alpha, scale=scale, zero_point=zero_point)
    
    if not cpu:
        elu = elu.cuda()
    
    result = elu(q_input)

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
        alpha = float(input_dict.get("alpha", 1.0))

        result = tf.nn.elu(input_tensor)
        result = tf.where(input_tensor <= 0, alpha * (tf.exp(input_tensor) - 1), result)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        "alpha": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    group_count = input_dict.get("group_count", 1)
    num_bits = input_dict.get("num_bits", 4)
    reduce_range = input_dict.get("reduce_range", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    min_val = torch.min(input_tensor)
    max_val = torch.max(input_tensor)

    if reduce_range:
        qmin = 0.0
        qmax = 2.0 ** (num_bits - 1) - 1.0
    else:
        qmin = - 2.0 ** (num_bits - 1)
        qmax = 2.0 ** (num_bits - 1) - 1.0

    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale

    zero_point = torch.clamp(zero_point, qmin, qmax)
    zero_point = torch.round(zero_point)

    if not cpu:
        scale = scale.cpu()
        zero_point = zero_point.cpu()

    return {"scale": scale.numpy(), "zero_point": zero_point.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        group_count = input_dict.get("group_count", 1)
        num_bits = input_dict.get("num_bits", 4)
        reduce_range = input_dict.get("reduce_range", False)

        input_tensor_min = tf.reduce_min(input_tensor)
        input_tensor_max = tf.reduce_max(input_tensor)

        if reduce_range:
           qmin = 0.0
           qmax = 2.0 ** (num_bits - 1) -1.0
        else:
           qmin = - 2.0 ** (num_bits - 1)
           qmax = 2.0 ** (num_bits - 1) - 1.0

        scale = (input_tensor_max - input_tensor_min) / (qmax - qmin)
        zero_point = qmin - input_tensor_min / scale

        zero_point = tf.clip_by_value(zero_point, qmin, qmax)
        zero_point = tf.round(zero_point)

        scale = scale.numpy()
        zero_point = zero_point.numpy()

    return {"scale": scale, "zero_point": zero_point}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "group_count": 1,
        "num_bits": 8,
        "reduce_range": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["scale"], tf_result["scale"], atol=A_TOL), "Scale results do not match"
    assert np.allclose(torch_result["zero_point"], tf_result["zero_point"], atol=A_TOL), "Zero point results do not match"

    print("Success")

if __name__ == "__main__":
    main()
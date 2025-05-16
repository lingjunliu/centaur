import numpy as np
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    lower = input_dict.get("lower", 1/8)
    upper = input_dict.get("upper", 1/3)
    inplace = input_dict.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    rrelu = torch.nn.RReLU(lower=lower, upper=upper, inplace=inplace)
    result = rrelu(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        lower = input_dict.get("lower", 1/8)
        upper = input_dict.get("upper", 1/3)

        mask = input_tensor < 0
        rand_val = tf.random.uniform(shape=input_tensor.shape, minval=lower, maxval=upper)
        result = tf.where(mask, input_tensor * rand_val, input_tensor)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        "lower": 1/8,
        "upper": 1/3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
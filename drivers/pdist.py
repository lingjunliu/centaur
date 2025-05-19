import numpy as np
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 2.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.pdist(input_tensor, p=p)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    p = input_dict.get("p", 2.0)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        n = tf.shape(input_tensor)[0]
        result_list = []
        for i in range(n):
            for j in range(i + 1, n):
                distance = tf.reduce_sum(tf.abs(input_tensor[i] - input_tensor[j]) ** p) ** (1/p)
                result_list.append(distance)
        result = tf.stack(result_list)
        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "p": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
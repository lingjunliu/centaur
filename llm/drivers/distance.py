import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.modules.distance import PairwiseDistance

    x1 = torch.tensor(input_dict["x1"])
    x2 = torch.tensor(input_dict["x2"])
    p = input_dict.get("p", 2.0)
    eps = input_dict.get("eps", 1e-06)

    if not cpu:
        x1 = x1.cuda()
        x2 = x2.cuda()

    pairwise_distance = PairwiseDistance(p=p, eps=eps)
    result = pairwise_distance(x1, x2)

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
        x1 = tf.constant(input_dict["x1"], dtype=tf.float32)
        x2 = tf.constant(input_dict["x2"], dtype=tf.float32)
        p = input_dict.get("p", 2.0)
        eps = input_dict.get("eps", 1e-06)

        x1_norm = tf.norm(x1, ord=p, axis=1)
        x2_norm = tf.norm(x2, ord=p, axis=1)

        diff = tf.abs(x1 - x2)
        distance = tf.pow(tf.reduce_sum(tf.pow(diff, p), axis=1), 1/p)
        
        distance = tf.clip_by_value(distance, clip_value_min=eps, clip_value_max=tf.float32.max)
        
        result = distance.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "x1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "x2": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "p": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
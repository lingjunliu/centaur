import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    total_count = torch.tensor(input_dict["total_count"])
    prob = torch.tensor(input_dict["prob"])

    if not cpu:
        total_count = total_count.cuda()
        prob = prob.cuda()

    result = torch.binomial(total_count.int(), prob.float())

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import tensorflow_probability as tfp
    tfd = tfp.distributions

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        total_count = tf.constant(input_dict["total_count"], dtype=tf.int32)
        prob = tf.constant(input_dict["prob"], dtype=tf.float32)

        binomial = tfd.Binomial(total_count=tf.cast(total_count, tf.float32), probs=prob)
        result = binomial.sample()

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "total_count": np.array([10, 5, 20], dtype=np.int32),
        "prob": np.array([0.5, 0.8, 0.3], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
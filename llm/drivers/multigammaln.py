import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    p = input_dict.get("p", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.special.multigammaln(input_tensor, p)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 1)
        p_tensor = tf.cast(p, dtype=tf.float32)

        def gammaln(x):
            return tf.math.lgamma(x)

        result = tf.reduce_sum(gammaln(input_tensor - 0.5 * tf.range(p_tensor, dtype=tf.float32)[:,None]), axis=0) + (0.25 * p_tensor * (p_tensor - 1.0) * tf.math.log(tf.constant(np.pi, dtype=tf.float32)))

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([3.0, 4.0, 5.0], dtype=np.float32),
        "p": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
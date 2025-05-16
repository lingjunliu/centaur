import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    total_count = torch.tensor(input_dict["total_count"])
    prob = torch.tensor(input_dict["prob"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        total_count = total_count.cuda()
        prob = prob.cuda()
    
    result = torch.binomial(total_count.long(), prob.float())
    
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
        input_tensor = tf.constant(input_dict["input"])
        total_count = tf.constant(input_dict["total_count"], dtype=tf.int32)
        prob = tf.constant(input_dict["prob"])

        result = tfd.Binomial(total_count=total_count, probs=prob).sample(tf.shape(input_tensor))
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "total_count": np.array([5, 5, 5, 5], dtype=np.int32),
        "prob": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
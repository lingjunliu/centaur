import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    trans_probs = torch.tensor(input_dict["trans_probs"])
    log_likelihood = torch.tensor(input_dict["log_likelihood"])
    alpha = input_dict.get("alpha", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        trans_probs = trans_probs.cuda()
        log_likelihood = log_likelihood.cuda()

    log_alpha = torch.empty(input_tensor.size(0))
    log_alpha[0] = torch.log(torch.tensor(alpha)) + input_tensor[0]

    for t in range(1, input_tensor.size(0)):
        log_forward_t = torch.logsumexp(log_alpha[t-1] + trans_probs[0], dim=0) + input_tensor[t]
        log_alpha[t] = log_forward_t

    result = log_alpha[-1]

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        trans_probs = tf.constant(input_dict["trans_probs"], dtype=tf.float32)
        log_likelihood = tf.constant(input_dict["log_likelihood"], dtype=tf.float32)
        alpha = input_dict.get("alpha", 1.0)

        T = input_tensor.shape[0]

        log_alpha = tf.TensorArray(dtype=tf.float32, size=T, dynamic_size=False)
        log_alpha = log_alpha.write(0, tf.math.log(alpha) + input_tensor[0])

        def body(t, log_alpha):
          log_forward_t = tf.math.reduce_logsumexp(tf.expand_dims(log_alpha.read(t-1), axis=0) + trans_probs, axis=1) + input_tensor[t]
          log_alpha = log_alpha.write(t, tf.reshape(log_forward_t, []))
          return t + 1, log_alpha

        t = tf.constant(1)
        _, log_alpha = tf.while_loop(lambda t, _: t < T, body, loop_vars=(t, log_alpha))

        log_alpha = log_alpha.stack()

        result = log_alpha[-1].numpy()

    return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "trans_probs": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "log_likelihood": np.array([0.5, 0.6, 0.7], dtype=np.float32),
        "alpha": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
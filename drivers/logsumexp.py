import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    # input
    x1 = torch.tensor(input["input"])
    x2 = input["dim"]
    keepdim = input.get("keepdim", False)

    if not cpu:
        x1 = x1.cuda()

    # output
    y = torch.logsumexp(x1, x2, keepdim=keepdim)

    if not cpu:
        y = y.cpu()

    return {"logsumexp": y.numpy()}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):        
        # input
        x1 = tf.constant(input["input"])
        x2 = input["dim"]
        keepdim = input.get("keepdim", False)

        # output
        y = tf.math.reduce_logsumexp(x1, axis=x2, keepdims=keepdim)
        
        return {"logsumexp": y.numpy()}
    
def main():
    input_dict = {
        'input': torch.randn(3, 3),
        'dim': 1,
        'keepdim': True
    }

    torch_output = torch_version(input_dict)
    tf_output = tensorflow_version(input_dict)

    print(torch_output["logsumexp"])
    print(tf_output["logsumexp"])

    assert np.allclose(torch_output["logsumexp"], tf_output["logsumexp"])
    print("equal")

if __name__ == "__main__":
    main()
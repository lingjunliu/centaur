import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.utils import fuse_conv_bn_weights

    conv_w = torch.tensor(input_dict["conv_w"])
    conv_b = torch.tensor(input_dict["conv_b"]) if "conv_b" in input_dict else None
    bn_rm = torch.tensor(input_dict["bn_rm"])
    bn_rv = torch.tensor(input_dict["bn_rv"])
    bn_w = torch.tensor(input_dict["bn_w"])
    bn_b = torch.tensor(input_dict["bn_b"])
    eps = input_dict.get("eps", 1e-5)

    if not cpu:
        conv_w = conv_w.cuda()
        if conv_b is not None:
            conv_b = conv_b.cuda()
        bn_rm = bn_rm.cuda()
        bn_rv = bn_rv.cuda()
        bn_w = bn_w.cuda()
        bn_b = bn_b.cuda()

    fused_w, fused_b = fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b, eps)

    if not cpu:
        fused_w = fused_w.cpu()
        fused_b = fused_b.cpu() if fused_b is not None else None

    return {"fused_w": fused_w.numpy(), "fused_b": fused_b.numpy() if fused_b is not None else None}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    conv_w = tf.constant(input_dict["conv_w"], dtype=tf.float32)
    if "conv_b" in input_dict:
        conv_b = tf.constant(input_dict["conv_b"], dtype=tf.float32)
    else:
        conv_b = tf.zeros(input_dict["bn_rm"].shape, dtype=tf.float32)

    bn_rm = tf.constant(input_dict["bn_rm"], dtype=tf.float32)
    bn_rv = tf.constant(input_dict["bn_rv"], dtype=tf.float32)
    bn_w = tf.constant(input_dict["bn_w"], dtype=tf.float32)
    bn_b = tf.constant(input_dict["bn_b"], dtype=tf.float32)
    eps = input_dict.get("eps", 1e-5)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        scale = bn_w / tf.sqrt(bn_rv + eps)
        shift = bn_b - bn_rm * scale

        fused_w = conv_w * tf.reshape(scale, (-1,) + (1,) * (len(conv_w.shape)-1))
        fused_b = conv_b * scale + shift

        fused_w = fused_w.numpy()
        fused_b = fused_b.numpy()

    return {"fused_w": fused_w, "fused_b": fused_b}

def main():
    A_TOL = 0.01

    input_data = {
        "conv_w": np.random.rand(16, 3, 3, 3).astype(np.float32),
        "conv_b": np.random.rand(16).astype(np.float32),
        "bn_rm": np.random.rand(16).astype(np.float32),
        "bn_rv": np.random.rand(16).astype(np.float32),
        "bn_w": np.random.rand(16).astype(np.float32),
        "bn_b": np.random.rand(16).astype(np.float32),
        "eps": 1e-5,
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["fused_w"], tf_result["fused_w"], atol=A_TOL), "Fused weights do not match"
    assert np.allclose(torch_result["fused_b"], tf_result["fused_b"], atol=A_TOL), "Fused biases do not match"

    print("Success")

if __name__ == "__main__":
    main()
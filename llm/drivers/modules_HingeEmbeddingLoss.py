import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    margin = input_dict.get("margin", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()

    hinge_loss = torch.nn.HingeEmbeddingLoss(margin=margin, reduction='mean')
    result = hinge_loss(input_tensor, target_tensor)

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
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        target_tensor = tf.convert_to_tensor(input_dict["target"], dtype=tf.float32)
        margin = input_dict.get("margin", 1.0)

        loss = tf.maximum(0.0, margin - tf.multiply(input_tensor, tf.cast(target_tensor, dtype=tf.float32)))
        result = tf.reduce_mean(loss)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.5, -0.8, 1.2], dtype=np.float32),
        "target": np.array([1, -1, 1], dtype=np.int32),
        "margin": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
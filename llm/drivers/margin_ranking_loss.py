import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input1 = torch.tensor(input_dict["input1"])
    input2 = torch.tensor(input_dict["input2"])
    target = torch.tensor(input_dict["target"])
    margin = input_dict.get("margin", 0.0)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input1 = input1.cuda()
        input2 = input2.cuda()
        target = target.cuda()

    result = torch.nn.functional.margin_ranking_loss(input1, input2, target, margin=margin, reduction=reduction)

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
        input1 = tf.constant(input_dict["input1"])
        input2 = tf.constant(input_dict["input2"])
        target = tf.constant(input_dict["target"])
        margin = input_dict.get("margin", 0.0)
        reduction = input_dict.get("reduction", 'mean')

        y = target
        x1 = input1
        x2 = input2
        
        loss = tf.maximum(0.0, margin - y * (x1 - x2))

        if reduction == 'none':
            pass
        elif reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(loss)
        else:
            raise ValueError(f"Invalid reduction option: {reduction}")

        result = loss.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "input2": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "target": np.array([1, -1, 1], dtype=np.float32),
        "margin": 0.5,
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
import numpy as np


def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input1_tensor = torch.tensor(input["input1"], dtype=torch.float32, requires_grad=True)
    input2_tensor = torch.tensor(input["input2"], dtype=torch.float32, requires_grad=True)
    target_tensor = torch.tensor(input["target"])

    margin = input.get("margin", 0.0)
    reduction = input.get("reduction", 'mean')

    # Apply to torch.nn.MarginRankingLoss
    criterion = torch.nn.MarginRankingLoss(margin=margin, reduction=reduction)
    loss = criterion(input1_tensor, input2_tensor, target_tensor)

    if not cpu:
        loss = loss.cpu()

    return {"margin_ranking_loss": float(loss.item())}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input1_tensor = tf.constant(input["input1"])
        input2_tensor = tf.constant(input["input2"])
        target_tensor = tf.constant(input["target"])
        
        margin = input.get("margin", 0.0)
        reduction = input.get("reduction", 'mean')

        # Compute the margin ranking loss using TensorFlow operations
        loss = tf.maximum(0.0, -target_tensor * (input1_tensor - input2_tensor) + margin)

        if reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(loss)
        elif reduction == 'none':
            pass  # no reduction means return the loss as is
        
        return {"margin_ranking_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input1": np.array([0.2, 0.4, 0.6], dtype=np.float32),
        "input2": np.array([0.1, 0.3, 0.5], dtype=np.float32),
        "target": np.array([1.0, -1.0, 1.0], dtype=np.float32),  # Target should be 1 or -1
        "margin": 0.0,
        "reduction": 'mean'
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert that prints either 'equal' or 'not equal'
    if np.isclose(torch_result["margin_ranking_loss"], tf_result["margin_ranking_loss"], rtol=1e-05):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()
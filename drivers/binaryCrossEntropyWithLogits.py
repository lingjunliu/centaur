import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    target_tensor = torch.tensor(input["target"]) 
    weight_tensor = torch.tensor(input.get("weight", None)) if input.get("weight", None) is not None else None
    reduction = input.get("reduction", 'mean')
    pos_weight_tensor = torch.tensor(input.get("pos_weight", None)) if input.get("pos_weight", None) is not None else None

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        if weight_tensor is not None:
            weight_tensor = weight_tensor.cuda()
        if pos_weight_tensor is not None:
            pos_weight_tensor = pos_weight_tensor.cuda()

    # Apply to torch.nn.functional.binary_cross_entropy_with_logits
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        input_tensor, target_tensor, weight=weight_tensor,
        reduction=reduction, pos_weight=pos_weight_tensor
    )

    if not cpu:
        loss = loss.cpu()

    return {"binary_cross_entropy_with_logits_loss": float(loss.item())}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        target_tensor = tf.constant(input["target"])
        weight_tensor = tf.constant(input.get("weight", None)) if input.get("weight", None) is not None else None
        pos_weight_tensor = tf.constant(input.get("pos_weight", None)) if input.get("pos_weight", None) is not None else None

        # Apply to TensorFlow equivalent
        if pos_weight_tensor is not None:
            loss = tf.nn.weighted_cross_entropy_with_logits(
                labels=target_tensor, logits=input_tensor, pos_weight=pos_weight_tensor
            )
        else:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=target_tensor, logits=input_tensor
            )

        if weight_tensor is not None:
            loss = loss * weight_tensor

        loss = tf.reduce_mean(loss)

        return {"binary_cross_entropy_with_logits_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "target": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),  # Ensure target is float type
        "weight": None,
        "reduction": 'mean',
        "pos_weight": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if torch_result["binary_cross_entropy_with_logits_loss"] == tf_result["binary_cross_entropy_with_logits_loss"]:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()

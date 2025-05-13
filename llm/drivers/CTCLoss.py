import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    input_lengths = torch.tensor(input_dict["input_lengths"])
    target_lengths = torch.tensor(input_dict["target_lengths"])
    blank = input_dict.get("blank", 0)
    reduction = input_dict.get("reduction", 'mean')
    zero_infinity = input_dict.get("zero_infinity", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()

    ctc_loss = torch.nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    result = ctc_loss(input_tensor.log_softmax(2), target, input_lengths, target_lengths)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    target = tf.constant(input_dict["target"], dtype=tf.int32)
    input_lengths = tf.constant(input_dict["input_lengths"], dtype=tf.int32)
    target_lengths = tf.constant(input_dict["target_lengths"], dtype=tf.int32)
    blank = input_dict.get("blank", 0)
    reduction = input_dict.get("reduction", 'mean')
    zero_infinity = input_dict.get("zero_infinity", False)

    if not cpu:
        device_string = "/gpu:0"
    else:
        device_string = "/cpu:0"

    with tf.device(device_string):
        max_target_length = np.max(input_dict["target_lengths"])
        target_matrix = np.zeros((input_data["target"].shape[0], max_target_length), dtype=np.int32)
        for i, target_len in enumerate(input_dict["target_lengths"]):
            target_matrix[i, :target_len] = input_dict["target"][i, :target_len]
        target = tf.constant(target_matrix, dtype=tf.int32)

        sparse_labels = tf.SparseTensor(
            indices=[[i, j] for i in range(input_data["target"].shape[0]) for j in range(input_dict["target_lengths"][i])],
            values=target.numpy().flatten()[:np.sum(input_dict["target_lengths"])],
            dense_shape=input_data["target"].shape
        )

        loss = tf.nn.ctc_loss(
            labels=sparse_labels,
            logits=tf.transpose(input_tensor, perm=[1, 0, 2]),
            sequence_length=input_lengths,
            blank_index=blank
        )

        if zero_infinity:
            mask = tf.math.is_inf(loss)
            loss = tf.where(mask, tf.zeros_like(loss), loss)
            
        if reduction == 'none':
            result = loss
        elif reduction == 'mean':
            result = tf.reduce_mean(loss)
        elif reduction == 'sum':
            result = tf.reduce_sum(loss)
        else:
            raise ValueError("Invalid reduction option")
            
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4).astype(np.float32),
        "target": np.array([[1, 2, 0], [1, 0, 0]], dtype=np.int32),
        "input_lengths": np.array([3, 3], dtype=np.int32),
        "target_lengths": np.array([2, 1], dtype=np.int32),
        "blank": 0,
        "reduction": "mean",
        "zero_infinity": False
    }

    batch_size = input_data["input"].shape[0]
    input_data["input_lengths"] = np.full(batch_size, input_data["input"].shape[1], dtype=np.int32)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
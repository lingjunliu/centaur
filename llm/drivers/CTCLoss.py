import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    targets = torch.tensor(input_dict["targets"])
    input_lengths = torch.tensor(input_dict["input_lengths"])
    target_lengths = torch.tensor(input_dict["target_lengths"])
    blank = input_dict.get("blank", 0)
    reduction = input_dict.get("reduction", 'mean')
    zero_infinity = input_dict.get("zero_infinity", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        targets = targets.cuda()
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()

    ctc_loss = torch.nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    result = ctc_loss(input_tensor, targets, input_lengths, target_lengths)

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
        input_tensor = tf.constant(input_dict["input"])
        targets = tf.constant(input_dict["targets"], dtype=tf.int32)
        input_lengths = tf.constant(input_dict["input_lengths"], dtype=tf.int32)
        target_lengths = tf.constant(input_dict["target_lengths"], dtype=tf.int32)
        blank = input_dict.get("blank", 0)
        reduction_str = input_dict.get("reduction", 'mean')
        zero_infinity = input_dict.get("zero_infinity", False)
        
        sparse_targets = tf.SparseTensor(
            indices=tf.stack([tf.range(tf.shape(targets)[0]), tf.range(tf.shape(targets)[1])], axis=1),
            values=targets.flatten(),
            dense_shape=targets.shape
        )

        loss = tf.nn.ctc_loss(
            labels=sparse_targets,
            logits=tf.transpose(input_tensor, perm=[1, 0, 2]),
            label_length=target_lengths,
            logit_length=input_lengths,
            blank_index=blank
        )

        if reduction_str == 'mean':
            result = tf.reduce_mean(loss)
        elif reduction_str == 'sum':
            result = tf.reduce_sum(loss)
        else:
            result = loss
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[0.1, 0.6, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.6, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.6, 0.1],
                            [0.6, 0.1, 0.1, 0.1, 0.1]]], dtype=np.float32),
        "targets": np.array([1, 2, 3], dtype=np.int32),
        "input_lengths": np.array([4], dtype=np.int32),
        "target_lengths": np.array([3], dtype=np.int32),
        "blank": 0,
        "reduction": "mean",
        "zero_infinity": False
    }

    input_data["input"] = np.transpose(input_data["input"], (1, 0, 2))
    input_data["input_lengths"] = np.array([input_data["input"].shape[0]], dtype=np.int32)
    input_data["target_lengths"] = np.array([input_data["target_lengths"]], dtype=np.int32)
    input_data["targets"] = np.array([input_data["targets"]], dtype=np.int32)
    input_data["input_lengths"] = np.array([input_data["input_lengths"]], dtype=np.int32)


    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
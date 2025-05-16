import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"], dtype=torch.int)
    input_lengths = torch.tensor(input_dict["input_lengths"], dtype=torch.int)
    target_lengths = torch.tensor(input_dict["target_lengths"], dtype=torch.int)
    blank = input_dict.get("blank", 0)
    reduction = input_dict.get("reduction", 'mean')
    zero_infinity = input_dict.get("zero_infinity", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()

    ctc_loss = torch.nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    loss = ctc_loss(input_tensor, target, input_lengths, target_lengths)
    
    if not cpu:
        loss = loss.cpu()
    
    return {"result": loss.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        target = tf.constant(input_dict["target"], dtype=tf.int32)
        input_lengths = tf.constant(input_dict["input_lengths"], dtype=tf.int32)
        target_lengths = tf.constant(input_dict["target_lengths"], dtype=tf.int32)
        blank = input_dict.get("blank", 0)
        reduction = input_dict.get("reduction", 'mean')
        zero_infinity = input_dict.get("zero_infinity", False)

        sparse_labels = tf.SparseTensor(
            indices=tf.stack([tf.range(tf.shape(target)[0]), tf.zeros(tf.shape(target)[0], dtype=tf.int64)], axis=1),
            values=tf.cast(tf.reshape(target, [-1]), dtype=tf.int32),
            dense_shape=[tf.shape(target)[0], 1]
        )
        
        loss = tf.nn.ctc_loss(
            labels=sparse_labels,
            logits=tf.transpose(input_tensor, [1, 0, 2]),
            label_length=target_lengths,
            logit_length=input_lengths,
            blank_index=blank
        )

        if reduction == 'none':
            pass
        elif reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(loss)

        if zero_infinity:
            loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)

        loss = loss.numpy()
    
    return {"result": loss}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.randn(1, 10, 5).astype(np.float32),
        "target": np.array([1, 2, 3], dtype=np.int32),
        "input_lengths": np.array([10], dtype=np.int32),
        "target_lengths": np.array([3], dtype=np.int32),
    }
    batch_size = input_data["input"].shape[0]
    input_data["input_lengths"] = np.array([input_data["input"].shape[1]] * batch_size, dtype=np.int32)
    input_data["target_lengths"] = np.array([input_data["target"].shape[0]] * batch_size, dtype=np.int32)
    

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
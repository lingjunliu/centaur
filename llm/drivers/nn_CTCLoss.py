import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    log_probs = torch.tensor(input_dict["log_probs"])
    targets = torch.tensor(input_dict["targets"])
    input_lengths = torch.tensor(input_dict["input_lengths"])
    target_lengths = torch.tensor(input_dict["target_lengths"])
    blank = input_dict.get("blank", 0)
    reduction = input_dict.get("reduction", 'mean')
    zero_infinity = input_dict.get("zero_infinity", False)

    if not cpu:
        log_probs = log_probs.cuda()
        targets = targets.cuda()
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()

    ctc_loss = torch.nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

    if not cpu:
        loss = loss.cpu()

    return {"result": loss.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    log_probs = tf.constant(input_dict["log_probs"])
    targets = tf.constant(input_dict["targets"], dtype=tf.int32)
    input_lengths = tf.constant(input_dict["input_lengths"], dtype=tf.int32)
    target_lengths = tf.constant(input_dict["target_lengths"], dtype=tf.int32)
    blank = input_dict.get("blank", 0)
    reduction = input_dict.get("reduction", 'mean')
    zero_infinity = input_dict.get("zero_infinity", False)
    N = input_dict["targets"].shape[0]


    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        sparse_labels = tf.keras.backend.ctc_label_dense_to_sparse(
            targets, target_lengths
        )
        
        loss = tf.compat.v1.nn.ctc_loss(
            labels=sparse_labels,
            inputs=tf.transpose(log_probs, [1, 0, 2]),
            sequence_length=input_lengths,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=True
        )

        if reduction == 'none':
            pass
        elif reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(loss)
        else:
            raise ValueError("Invalid reduction option.")

        if zero_infinity:
            loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)

        loss = loss.numpy()

    return {"result": loss}

def main():
    A_TOL = 0.01

    T = 5
    N = 2
    C = 3
    S = 3
    S_min = 1

    input_data = {
        "log_probs": np.random.randn(T, N, C).astype(np.float32),
        "targets": np.random.randint(low=1, high=C, size=(N, S), dtype=np.int32),
        "input_lengths": np.full((N,), fill_value=T, dtype=np.int32),
        "target_lengths": np.random.randint(low=S_min, high=S, size=(N,), dtype=np.int32),
        "blank": 0,
        "reduction": 'mean',
        "zero_infinity": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    np.testing.assert_allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()
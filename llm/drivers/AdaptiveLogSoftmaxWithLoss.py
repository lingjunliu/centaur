import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    n_classes = input_dict["n_classes"]
    cutoffs = input_dict.get("cutoffs", [5000, 10000, 20000])
    div_value = input_dict.get("div_value", 4.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()

    adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(
        in_features=input_tensor.shape[-1],
        n_classes=n_classes,
        cutoffs=cutoffs,
        div_value=div_value,
    )

    if not cpu:
        adaptive_log_softmax = adaptive_log_softmax.cuda()

    loss, output = adaptive_log_softmax(input_tensor, target)

    if not cpu:
        loss = loss.cpu()
        output = output.cpu()

    return {"loss": loss.detach().numpy(), "output": output.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        target = tf.constant(input_dict["target"], dtype=tf.int64)
        n_classes = input_dict["n_classes"]
        cutoffs = input_dict.get("cutoffs", [5000, 10000, 20000])
        div_value = input_dict.get("div_value", 4.0)

        batch_size = tf.shape(input_tensor)[0]
        input_dim = tf.shape(input_tensor)[1]

        def adaptive_softmax_loss(inputs, labels, num_classes, cutoffs, div_value):
            
            def build_head(inputs, div_value, num_classes):
              head_size = num_classes // int(div_value) if num_classes > div_value else num_classes
              return tf.keras.layers.Dense(head_size)(inputs)
            
            head_logits = build_head(inputs, div_value, num_classes)
            head_log_probs = tf.nn.log_softmax(head_logits)
            
            losses = []
            
            # Head Loss
            head_size = num_classes // int(div_value) if num_classes > div_value else num_classes
            head_labels = tf.clip_by_value(labels // (cutoffs[0] // int(div_value)), 0, head_size - 1)
            head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=head_labels, logits=head_logits)
            losses.append(tf.reduce_mean(head_loss))

            # Cutoff Losses
            prev_cutoff = 0
            for i, cutoff in enumerate(cutoffs):
                cutoff_idx = tf.where((labels >= prev_cutoff) & (labels < cutoff))
                cutoff_inputs = tf.gather_nd(inputs, cutoff_idx)
                cutoff_labels = tf.gather_nd(labels, cutoff_idx)

                if tf.shape(cutoff_labels)[0] > 0:
                    cutoff_logits = tf.keras.layers.Dense(cutoff - prev_cutoff)(cutoff_inputs)
                    cutoff_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cutoff_labels - prev_cutoff, logits=cutoff_logits)
                    losses.append(tf.reduce_mean(cutoff_loss))
                prev_cutoff = cutoff

            # Remaining Loss
            if cutoffs[-1] < n_classes:
                remaining_idx = tf.where(labels >= cutoffs[-1])
                remaining_inputs = tf.gather_nd(inputs, remaining_idx)
                remaining_labels = tf.gather_nd(labels, remaining_idx)

                if tf.shape(remaining_labels)[0] > 0:
                    remaining_logits = tf.keras.layers.Dense(n_classes - cutoffs[-1])(remaining_inputs)
                    remaining_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=remaining_labels - cutoffs[-1], logits=remaining_logits)
                    losses.append(tf.reduce_mean(remaining_loss))

            loss = tf.add_n(losses)

            output = tf.zeros(shape=(batch_size, num_classes))

            return loss, output

        loss, output = adaptive_softmax_loss(input_tensor, target, n_classes, cutoffs, div_value)
    
    return {"loss": loss.numpy(), "output": output.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(10, 128).astype(np.float32),
        "target": np.random.randint(0, 10000, size=(10,)).astype(np.int64),
        "n_classes": 10000,
        "cutoffs": [3000, 7000],
        "div_value": 2.0,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["loss"], tf_result["loss"], atol=A_TOL), "Losses do not match"
    print("Success")

if __name__ == "__main__":
    main()
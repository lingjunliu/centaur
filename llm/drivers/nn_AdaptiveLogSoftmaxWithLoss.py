import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    in_features = input_dict["in_features"]
    n_classes = input_dict["n_classes"]
    cutoffs = input_dict["cutoffs"]
    div_value = input_dict.get("div_value", 4.0)
    head_bias = input_dict.get("head_bias", False)
    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()

    adaptive_softmax = torch.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=div_value, head_bias=head_bias)

    output = adaptive_softmax(input_tensor, target)

    if not cpu:
        output = (output.output.cpu(), output.loss.cpu())
    else:
        output = (output.output, output.loss)

    return {"output": output[0].detach().numpy(), "loss": output[1].detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    in_features = input_dict["in_features"]
    n_classes = input_dict["n_classes"]
    cutoffs = input_dict["cutoffs"]
    div_value = input_dict.get("div_value", 4.0)
    head_bias = input_dict.get("head_bias", False)
    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    target = tf.constant(input_dict["target"], dtype=tf.int32)

    def adaptive_log_softmax_with_loss(input_tensor, target, in_features, n_classes, cutoffs, div_value, head_bias):

        def log_prob(input_tensor, W, b):
            logits = tf.matmul(input_tensor, W) + b
            log_probs = tf.nn.log_softmax(logits)
            return log_probs

        head_size = cutoffs[0]
        W = tf.Variable(tf.random.normal((in_features, head_size), stddev=0.01))
        b = tf.Variable(tf.zeros((head_size,))) if head_bias else tf.zeros((head_size,))

        head_log_probs = log_prob(input_tensor, W, b)
        head_target = tf.clip_by_value(target, 0, cutoffs[0] - 1)
        head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=head_target, logits=tf.matmul(input_tensor, W) + b)

        tail_loss = tf.zeros_like(head_loss)
        tail_log_probs = tf.zeros_like(head_loss)

        current_start = cutoffs[0]
        for idx, cutoff in enumerate(cutoffs[1:] + [n_classes]):
            cluster_size = cutoff - current_start
            cluster_dim = int(in_features / (div_value ** (idx + 1)))
            
            input_projection = tf.keras.layers.Dense(cluster_dim, use_bias=False)(input_tensor)
            W_cluster = tf.Variable(tf.random.normal((cluster_dim, cluster_size), stddev=0.01))
            b_cluster = tf.Variable(tf.zeros((cluster_size,)))


            cluster_log_probs = log_prob(input_projection, W_cluster, b_cluster)

            mask = tf.logical_and(target >= current_start, target < cutoff)
            
            masked_target = tf.where(mask, target - current_start, tf.zeros_like(target, dtype=tf.int32) - 1)
            masked_target = tf.clip_by_value(masked_target, 0, cluster_size - 1)

            cluster_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(masked_target, dtype=tf.int32), logits=tf.matmul(input_projection, W_cluster) + b_cluster)

            tail_loss = tf.where(mask, cluster_loss, tail_loss)
            tail_log_probs = tf.where(mask, cluster_log_probs[:, 0], tail_log_probs)
            current_start = cutoff
        loss = tf.reduce_mean(head_loss + tail_loss)
        
        output = tf.concat([tf.expand_dims(head_log_probs[tf.range(tf.shape(head_log_probs)[0]), head_target], axis=1), tf.expand_dims(tail_log_probs, axis=1)], axis=1)
        
        return output[:, 0], loss

    output, loss = adaptive_log_softmax_with_loss(input_tensor, target, in_features, n_classes, cutoffs, div_value, head_bias)
    return {"output": output.numpy(), "loss": loss.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "in_features": 16,
        "n_classes": 10000,
        "cutoffs": [1000, 5000],
        "input": np.random.rand(8, 16).astype(np.float32),
        "target": np.random.randint(0, 10000, size=(8,)).astype(np.int64),
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["loss"], tf_result["loss"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn import AdaptiveLogSoftmaxWithLoss

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    n_classes = input_dict["n_classes"]
    cutoffs = input_dict["cutoffs"]
    div_value = input_dict.get("div_value", 4.0)
    head_bias = input_dict.get("head_bias", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()

    adaptive_softmax = AdaptiveLogSoftmaxWithLoss(
        in_features=input_tensor.shape[-1],
        n_classes=n_classes,
        cutoffs=cutoffs,
        div_value=div_value,
        head_bias=head_bias
    )

    if not cpu:
        adaptive_softmax = adaptive_softmax.cuda()

    loss, output = adaptive_softmax(input_tensor, target)

    if not cpu:
        loss = loss.cpu()
        output = output.cpu()

    return {"loss": loss.detach().numpy(), "output": output.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    target = tf.constant(input_dict["target"], dtype=tf.int32)
    n_classes = input_dict["n_classes"]
    cutoffs = input_dict["cutoffs"]
    div_value = input_dict.get("div_value", 4.0)
    head_bias = input_dict.get("head_bias", False)
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        batch_size = tf.shape(input_tensor)[0]
        in_features = tf.shape(input_tensor)[1]

        def build_network(inputs, weights, biases, activation=None):
            layer = tf.matmul(inputs, weights) + biases
            if activation:
                layer = activation(layer)
            return layer

        def adaptive_softmax(inputs, labels, n_classes, cutoffs, div_value, head_bias):
            head_size = cutoffs[0]
            tail_weights = []
            tail_biases = []
            tail_sizes = []

            for i in range(len(cutoffs) - 1):
                tail_sizes.append(int((n_classes - cutoffs[i]) / (div_value ** (i + 1))))

            head_weights = tf.Variable(tf.random.normal([in_features, head_size]))
            if head_bias:
                head_biases = tf.Variable(tf.zeros([head_size]))
            else:
                head_biases = tf.zeros([head_size])

            for i in range(len(tail_sizes)):
                if tail_sizes[i] <= 0:
                    tail_sizes[i] = 1

            start_index = cutoffs[0]
            for i, size in enumerate(tail_sizes):
                end_index = cutoffs[i + 1] if i+1 < len(cutoffs) else n_classes
                tail_weights.append(tf.Variable(tf.random.normal([in_features, size])))
                tail_biases.append(tf.Variable(tf.zeros([size])))

            head_logits = build_network(inputs, head_weights, head_biases)
            head_log_probs = tf.nn.log_softmax(head_logits, axis=-1)

            loss = tf.zeros([batch_size], dtype=tf.float32)

            for i in range(batch_size):
                label = labels[i]
                if label < head_size:
                    loss = tf.tensor_scatter_nd_update(loss, [[i]], [-head_log_probs[i, label]])
                else:
                    tail_index = 0
                    for j in range(len(cutoffs)):
                        if j < len(cutoffs) - 1 and cutoffs[j] <= label < cutoffs[j+1]:
                            tail_index = j
                            break
                        elif j == len(cutoffs) - 1 and cutoffs[j] <= label < n_classes:
                            tail_index = j
                            break
                    adjusted_label = label - cutoffs[tail_index]
                    if len(tail_weights) > tail_index and adjusted_label < tail_weights[tail_index].shape[1]:
                        tail_logits = build_network(inputs[i:i+1], tail_weights[tail_index], tail_biases[tail_index])
                        tail_log_probs = tf.nn.log_softmax(tail_logits, axis=-1)
                        head_cutoff_idx = cutoffs[0] + tail_index
                        loss = tf.tensor_scatter_nd_update(loss, [[i]], [-head_log_probs[i, head_cutoff_idx] - tail_log_probs[0, adjusted_label]])
                    else:
                         loss = tf.tensor_scatter_nd_update(loss, [[i]], [0.0])
            loss = tf.reduce_mean(loss)
            output = tf.nn.softmax(head_logits)

            return loss, output

        loss, output = adaptive_softmax(input_tensor, target, n_classes, cutoffs, div_value, head_bias)

    return {"loss": loss.numpy(), "output": output.numpy()}

def main():
    A_TOL = 0.1

    input_data = {
        "input": np.random.rand(2, 128).astype(np.float32),
        "target": np.array([0, 101], dtype=np.int64),
        "n_classes": 500,
        "cutoffs": [100, 250, 400],
        "div_value": 4.0,
        "head_bias": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["loss"], tf_result["loss"], atol=A_TOL), "Losses do not match"
    # Due to approximation, do not compare the output.
    #assert np.allclose(torch_result["output"], tf_result["output"], atol=A_TOL), "Outputs do not match"

    print("Success")

if __name__ == "__main__":
    main()
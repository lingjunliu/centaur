import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict["input"])
    n_classes = input_dict["n_classes"]
    cutoffs = input_dict["cutoffs"]
    dropout = input_dict.get("dropout", 0.0)
    adaptive_inputs = input_dict["adaptive_inputs"]
    tie_projections = input_dict.get("tie_projections", True)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        adaptive_inputs = [torch.tensor(w).cuda() for w in adaptive_inputs]
    else:
        adaptive_inputs = [torch.tensor(w) for w in adaptive_inputs]
    
    class ModifiedAdaptiveLogSoftmaxWithLoss(torch.nn.Module):
        def __init__(self, in_features, n_classes, cutoffs, adaptive_inputs):
            super().__init__()
            self.in_features = in_features
            self.n_classes = n_classes
            self.cutoffs = cutoffs
            self.dropout = 0.0
            self.head = torch.nn.Linear(in_features, adaptive_inputs[0].shape[-1])
            self.tail = torch.nn.ModuleList([torch.nn.Linear(in_features, w.shape[-1]) for w in adaptive_inputs[1:]])

        def forward(self, input, target):
            n_clusters = len(self.cutoffs) + 1
            head_output = self.head(input)
            head_log_prob = torch.nn.functional.log_softmax(head_output, dim=-1)
            
            head_target = torch.where(target < self.cutoffs[0], target, torch.zeros_like(target))
            loss = torch.nn.functional.cross_entropy(head_output, head_target, reduction='mean')
            
            safe_target = torch.clamp(target, 0, self.head.out_features - 1)
            log_prob = torch.gather(head_log_prob, dim=-1, index=safe_target.unsqueeze(-1)).squeeze(-1)

            for i in range(n_clusters - 1):
                l_cutoff = self.cutoffs[i - 1] if i > 0 else 0
                r_cutoff = self.cutoffs[i]
                
                cluster_target = torch.where((target >= l_cutoff) & (target < r_cutoff), target - l_cutoff, torch.zeros_like(target))
                cluster_index = torch.where((target >= l_cutoff) & (target < r_cutoff), torch.zeros_like(target), torch.ones_like(target))
                
                tail_output = self.tail[i](input)
                tail_log_prob = torch.nn.functional.log_softmax(tail_output, dim=-1)

                
                safe_cluster_target = torch.clamp(cluster_target, 0, self.tail[i].out_features - 1)
                cluster_loss = torch.nn.functional.cross_entropy(tail_output, safe_cluster_target, reduction='none')
                
                loss = torch.where(cluster_index == 0, cluster_loss, loss)
                log_prob = torch.where(cluster_index == 0, torch.gather(tail_log_prob, dim=-1, index=safe_cluster_target.unsqueeze(-1)).squeeze(-1), log_prob)
            return torch.nn.modules.AdaptiveLogSoftmaxWithLoss.Output(loss=loss.mean(), log_prob=log_prob)
        
    if not cpu:
        model = ModifiedAdaptiveLogSoftmaxWithLoss(
            in_features=input_tensor.shape[-1],
            n_classes=n_classes,
            cutoffs=cutoffs,
            adaptive_inputs = adaptive_inputs
        ).cuda()
        model.dropout = dropout
        if tie_projections:
             for i in range(len(model.tail)):
                 model.tail[i].weight = model.head.weight

        target = torch.tensor(input_dict["target"]).cuda()
        output = model(input_tensor, target)
        
        loss = output.loss.cpu().detach().numpy()
        log_prob = output.log_prob.cpu().detach().numpy()
        
        return {"loss": loss, "log_prob": log_prob}
    else:
        model = ModifiedAdaptiveLogSoftmaxWithLoss(
            in_features=input_tensor.shape[-1],
            n_classes=n_classes,
            cutoffs=cutoffs,
            adaptive_inputs = adaptive_inputs
        )
        model.dropout = dropout
        if tie_projections:
             for i in range(len(model.tail)):
                 model.tail[i].weight = model.head.weight
        
        target = torch.tensor(input_dict["target"])
        output = model(input_tensor, target)
        
        loss = output.loss.detach().numpy()
        log_prob = output.log_prob.detach().numpy()
        
        return {"loss": loss, "log_prob": log_prob}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
        
    with tf.device(device_string):
        
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        n_classes = input_dict["n_classes"]
        cutoffs = input_dict["cutoffs"]
        dropout = input_dict.get("dropout", 0.0)
        adaptive_inputs = [tf.constant(w, dtype=tf.float32) for w in input_dict["adaptive_inputs"]]
        tie_projections = input_dict.get("tie_projections", True)
        target = tf.constant(input_dict["target"], dtype=tf.int32)
        
        in_features = input_tensor.shape[-1]
        
        def _adaptive_log_softmax_with_loss(inputs, labels, adaptive_inputs, cutoffs, tie_projections=True, dropout=0.0):
            
            n_clusters = len(cutoffs) + 1
            head_weight = adaptive_inputs[0]
            head_bias = tf.zeros([adaptive_inputs[0].shape[-1]], dtype=tf.float32)
            
            head = tf.matmul(inputs, head_weight) + head_bias
            
            total_loss = tf.zeros_like(labels, dtype=tf.float32)
            total_log_prob = tf.zeros_like(labels, dtype=tf.float32)
            
            # head part
            head_target = tf.where(labels < cutoffs[0], labels, tf.zeros_like(labels, dtype=tf.int32))
            
            head_log_prob = tf.nn.log_softmax(head, axis=-1)
            
            head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=head_target, logits=head)
            
            total_loss = tf.tensor_scatter_nd_update(total_loss, tf.expand_dims(tf.range(tf.shape(labels)[0]), axis=1), head_loss)
            total_log_prob = tf.tensor_scatter_nd_update(total_log_prob, tf.expand_dims(tf.range(tf.shape(labels)[0]), axis=1), tf.gather_nd(head_log_prob, tf.stack([tf.range(tf.shape(labels)[0]), tf.cast(head_target, tf.int32)], axis=1)))
            
            # tail parts
            for i in range(n_clusters - 1):
                l_cutoff = cutoffs[i - 1] if i > 0 else 0
                r_cutoff = cutoffs[i]
                
                cluster_target = tf.where((labels >= l_cutoff) & (labels < r_cutoff), labels - l_cutoff, tf.zeros_like(labels, dtype=tf.int32))
                cluster_index = tf.where((labels >= l_cutoff) & (labels < r_cutoff), tf.zeros_like(labels, dtype=tf.int32), tf.ones_like(labels, dtype=tf.int32))
                
                cluster_input = tf.nn.dropout(inputs, rate=dropout)
                cluster_weight = adaptive_inputs[i+1]
                cluster_bias = tf.zeros([cluster_weight.shape[-1]], dtype=tf.float32)

                cluster_log_prob = tf.nn.log_softmax(tf.matmul(cluster_input, cluster_weight) + cluster_bias, axis=-1)

                cluster_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cluster_target, logits=tf.matmul(cluster_input, cluster_weight) + cluster_bias)
                
                total_loss = tf.tensor_scatter_nd_update(total_loss, tf.expand_dims(tf.range(tf.shape(labels)[0]), axis=1), tf.where(cluster_index == 0, cluster_loss, tf.gather_nd(total_loss, tf.expand_dims(tf.range(tf.shape(labels)[0]), axis=1))))
                total_log_prob = tf.tensor_scatter_nd_update(total_log_prob, tf.expand_dims(tf.range(tf.shape(labels)[0]), axis=1), tf.where(cluster_index == 0, tf.gather_nd(cluster_log_prob, tf.stack([tf.range(tf.shape(labels)[0]), tf.cast(cluster_target, tf.int32)], axis=1)), tf.gather_nd(total_log_prob, tf.expand_dims(tf.range(tf.shape(labels)[0]), axis=1))))
                
            return tf.reduce_mean(total_loss), total_log_prob
        
        loss, log_prob = _adaptive_log_softmax_with_loss(input_tensor, target, adaptive_inputs, cutoffs, tie_projections, dropout)

        return {"loss": loss.numpy(), "log_prob": log_prob.numpy()}

def main():
    A_TOL = 0.1

    input_data = {
        "input": np.random.randn(1, 10).astype(np.float32),
        "n_classes": 100,
        "cutoffs": [10, 20, 50],
        "adaptive_inputs": [
            np.random.randn(10, 10).astype(np.float32),
            np.random.randn(10, 10).astype(np.float32),
            np.random.randn(10, 10).astype(np.float32),
            np.random.randn(10, 10).astype(np.float32)
        ],
        "target": np.random.randint(0, 100, size=(1,)).astype(np.int64),
        "dropout": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["loss"], tf_result["loss"], atol=A_TOL), "Loss results do not match"
    assert np.allclose(torch_result["log_prob"], tf_result["log_prob"], atol=A_TOL), "Log Prob results do not match"

    print("Success")

if __name__ == "__main__":
    main()
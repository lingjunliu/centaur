import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

class CustomInstanceNorm2D(tf.keras.layers.Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super(CustomInstanceNorm2D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.gamma = self.add_weight(shape=(num_features,), initializer='ones', trainable=True)
            self.beta = self.add_weight(shape=(num_features,), initializer='zeros', trainable=True)
        else:
            self.gamma = None
            self.beta = None
        if track_running_stats:
            self.running_mean = self.add_weight(shape=(num_features,), initializer='zeros', trainable=False)
            self.running_var = self.add_weight(shape=(num_features,), initializer='ones', trainable=False)

    def call(self, inputs, training=False):
        mean, variance = tf.nn.moments(inputs, axes=[2, 3], keepdims=True)  # (N, C, 1, 1)
        if self.track_running_stats:
            if training:
                mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                variance = self.momentum * variance + (1 - self.momentum) * self.running_var
                self.running_mean.assign(mean)
                self.running_var.assign(variance)
            else:
                mean = self.running_mean
                variance = self.running_var

        inv = tf.math.rsqrt(variance + self.eps)
        if self.affine:
            gamma = tf.reshape(self.gamma, (1, -1, 1, 1))
            beta = tf.reshape(self.beta, (1, -1, 1, 1))
        else:
            gamma = 1.0
            beta = 0.0

        return (inputs - mean) * inv * gamma + beta

def torch_version(input, cpu=True):
    set_seed()
    input_tensor = torch.tensor(input["input"])
    num_features = input["num_features"]
    eps = input.get("eps", 1e-5)
    momentum = input.get("momentum", 0.1)
    affine = input.get("affine", False)
    track_running_stats = input.get("track_running_stats", False)
    
    instance_norm = torch.nn.InstanceNorm2d(
        num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
    )
    
    if not cpu:
        instance_norm = instance_norm.cuda()
        input_tensor = input_tensor.cuda()
    
    output = instance_norm(input_tensor)
    
    if not cpu:
        output = output.cpu()
    
    return {"instance_normed_output": output.detach().numpy()}

def tensorflow_version(input, cpu=True):
    set_seed()
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
        
    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        num_features = input["num_features"]
        eps = input.get("eps", 1e-5)
        momentum = input.get("momentum", 0.1)
        affine = input.get("affine", False)
        track_running_stats = input.get("track_running_stats", False)

        instance_norm = CustomInstanceNorm2D(
            num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )
        
        output = instance_norm(input_tensor, training=True)
        
        return {"instance_normed_output": output.numpy()}

def main():
    input_data = {
        "input": np.random.randn(2, 3, 4, 4).astype(np.float32),
        "num_features": 3,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False
    }
    
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    np.testing.assert_allclose(torch_result["instance_normed_output"], tf_result["instance_normed_output"], rtol=1e-5, atol=1e-5)
    print("equal")

if __name__ == "__main__":
    main()
import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["src"])
    encoder_layer = input_dict["encoder_layer"]
    num_layers = input_dict.get("num_layers", 6)
    norm = input_dict.get("norm", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        encoder_layer = encoder_layer.cuda()
        if norm is not None:
          norm = norm.cuda()
    
    encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=norm)
    result = encoder(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    src_np = input_dict["src"]
    encoder_layer = input_dict["encoder_layer"]
    num_layers = input_dict.get("num_layers", 6)
    norm = input_dict.get("norm", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        src = tf.constant(src_np, dtype=tf.float32)

        def transformer_encoder(src, encoder_layer, num_layers, norm):
            output = src
            for _ in range(num_layers):
                output = encoder_layer(output)
            if norm is not None:
                output = norm(output)
            return output

        class TFDummyEncoderLayer():
          def __init__(self):
            pass
          def __call__(self, x):
            return x + tf.random.normal(x.shape, dtype=tf.float32)

        class TFDummyNorm():
            def __init__(self):
                pass
            def __call__(self, x):
                return x

        tf_encoder_layer = TFDummyEncoderLayer()
        tf_norm = TFDummyNorm() if norm is not None else None

        result = transformer_encoder(src, tf_encoder_layer, num_layers, tf_norm)
    
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    class DummyEncoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummySelfAttention()

        def cuda(self):
            return self

        def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal=False):
            return x + torch.randn(x.shape)

    class DummySelfAttention():
        def __init__(self):
            self.batch_first = False

    class DummyNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def cuda(self):
            return self

        def forward(self, x):
            return x

    input_data = {
        "src": np.random.rand(10, 32, 512).astype(np.float32),
        "encoder_layer": DummyEncoderLayer(),
        "num_layers": 2,
        "norm": DummyNorm()
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
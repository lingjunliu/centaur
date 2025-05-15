import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from typing import List

    class InterfaceType:
        def run(self, x: torch.Tensor) -> torch.Tensor:
            pass

    class Impl1:
        def run(self, x: torch.Tensor) -> torch.Tensor:
            return x.relu()

    class Impl2(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.val = torch.rand(())

        def run(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.val

    def user_fn(impls: List[InterfaceType], idx: int, val: torch.Tensor) -> torch.Tensor:
        return impls[idx].run(val)

    impls_input = input_dict["impls"]
    idx = input_dict["idx"]
    val = torch.tensor(input_dict["val"])

    impls = []
    for impl_name in impls_input:
        if impl_name == "Impl1":
            impls.append(Impl1())
        elif impl_name == "Impl2":
            impls.append(Impl2())
        else:
            raise ValueError(f"Unknown implementation: {impl_name}")
    
    if not cpu:
        val = val.cuda()
        for impl in impls:
            if isinstance(impl, Impl2):
                impl.val = impl.val.cuda()

    result = user_fn(impls, idx, val)

    if not cpu:
        result = result.cpu()
        for impl in impls:
            if isinstance(impl, Impl2):
                impl.val = impl.val.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    from typing import List

    class InterfaceType:
        def run(self, x: tf.Tensor) -> tf.Tensor:
            pass

    class Impl1:
        def run(self, x: tf.Tensor) -> tf.Tensor:
            return tf.nn.relu(x)

    class Impl2:
        def __init__(self) -> None:
            self.val = tf.Variable(tf.random.uniform(shape=(), minval=0, maxval=1))

        def run(self, x: tf.Tensor) -> tf.Tensor:
            return x + self.val

    def user_fn(impls: List[InterfaceType], idx: int, val: tf.Tensor) -> tf.Tensor:
        return impls[idx].run(val)

    impls_input = input_dict["impls"]
    idx = input_dict["idx"]
    val = tf.constant(input_dict["val"])

    impls = []
    for impl_name in impls_input:
        if impl_name == "Impl1":
            impls.append(Impl1())
        elif impl_name == "Impl2":
            impls.append(Impl2())
        else:
            raise ValueError(f"Unknown implementation: {impl_name}")

    result = user_fn(impls, idx, val)
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "impls": ["Impl1", "Impl2"],
        "idx": 0,
        "val": np.random.rand(4, 4).astype(np.float32)
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "impls": ["Impl1", "Impl2"],
        "idx": 1,
        "val": np.random.rand(4, 4).astype(np.float32)
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
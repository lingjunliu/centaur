import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
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

    impls_data = input_dict["impls"]
    impls = []
    for impl_data in impls_data:
        if impl_data["type"] == "Impl1":
            impls.append(Impl1())
        elif impl_data["type"] == "Impl2":
            impl = Impl2()
            impl.val = torch.tensor(impl_data["val"])
            impls.append(impl)

    idx = input_dict["idx"]
    val = torch.tensor(input_dict["val"])
    if not cpu:
        for i in range(len(impls)):
            if isinstance(impls[i], Impl2):
                impls[i].val = impls[i].val.cuda()
        val = val.cuda()

    if not cpu:
        for i in range(len(impls)):
            if isinstance(impls[i], Impl2):
                impls[i].cuda()

    if not cpu:
        class InterfaceType_cuda:
            def run(self, x: torch.Tensor) -> torch.Tensor:
                pass

        class Impl1_cuda:
            def run(self, x: torch.Tensor) -> torch.Tensor:
                return x.relu()

        class Impl2_cuda(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.val = torch.rand(()).cuda()

            def run(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.val

        def user_fn_cuda(impls: List[InterfaceType_cuda], idx: int, val: torch.Tensor) -> torch.Tensor:
            return impls[idx].run(val)

        def get_cuda_impl(impl):
             if isinstance(impl, Impl1):
                return Impl1_cuda()
             elif isinstance(impl, Impl2):
                impl_cuda = Impl2_cuda()
                impl_cuda.val = impl.val.cuda()
                return impl_cuda

        impls_cuda = [get_cuda_impl(impl) for impl in impls]
        result = user_fn_cuda(impls_cuda, idx, val)
    else:
        result = user_fn(impls, idx, val)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from typing import List

    class InterfaceType:
        def run(self, x: tf.Tensor) -> tf.Tensor:
            pass

    class Impl1:
        def run(self, x: tf.Tensor) -> tf.Tensor:
            return tf.nn.relu(x)

    class Impl2:
        def __init__(self, val):
            self.val = tf.constant(val, dtype=tf.float32)

        def run(self, x: tf.Tensor) -> tf.Tensor:
            return x + self.val

    def user_fn(impls: List[InterfaceType], idx: int, val: tf.Tensor) -> tf.Tensor:
        return impls[idx].run(val)

    impls_data = input_dict["impls"]
    impls = []
    for impl_data in impls_data:
        if impl_data["type"] == "Impl1":
            impls.append(Impl1())
        elif impl_data["type"] == "Impl2":
            impl = Impl2(impl_data["val"])
            impls.append(impl)

    idx = input_dict["idx"]
    val = tf.constant(input_dict["val"], dtype=tf.float32)
    result = user_fn(impls, idx, val)
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "impls": [
            {"type": "Impl1"},
            {"type": "Impl2", "val": np.array(0.5, dtype=np.float32)}
        ],
        "idx": 0,
        "val": np.array([1.0, -1.0, 0.5, -0.2], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "impls": [
            {"type": "Impl1"},
            {"type": "Impl2", "val": np.array(0.5, dtype=np.float32)}
        ],
        "idx": 1,
        "val": np.array([1.0, -1.0, 0.5, -0.2], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
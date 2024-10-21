import time
import torch as th
import tensorly as tl
tl.set_backend("pytorch")
th.set_default_dtype(th.float64)
th.manual_seed(42)


def mul_2D(w_, input_):
    K, E, H, D =  w_.shape
    weight_res =  w_.reshape((K, E, H*D)).swapaxes(-2, -1)
    weight_res = weight_res.permute(1, 0, 2)
    weight_res = weight_res.reshape((H*D, -1))
    print(f"Input shape: {input_.shape}")
    print(f"Wright shape: {weight_res.shape}")
    Y = input_ @ weight_res
    Y = Y.reshape((7, 5, K, 768)).permute(2, 0, 1, 3)
    print(f"2D output Y shape: {Y.shape}")
    return Y


def mul_3D(w_, input_):
    K, E, H, D = w_.shape
    x_res = input_.reshape((7, 5, H, D))
    decomp = tl.decomposition.CP(rank=4, normalize_factors=False,
        verbose=False, init="random", tol=1e-24, random_state=42
    )
    weight_terf = w_.reshape((K*E, H, D))
    _, (A2, A3, A4) = decomp.fit_transform(weight_terf)
    recons_weight = tl.cp_to_tensor((None, (A2, A3, A4)))
    print(th.allclose(recons_weight, weight_terf))
    xp = x_res.unsqueeze(0)
    A2 = preprocess(A2)
    A3 = preprocess(A3)
    A4 = preprocess(A4)
    inter_1 = xp @ A4.swapaxes(-2, -1)
    inter_1 = inter_1.squeeze(-1)
    inter_2 = inter_1 @ A3.squeeze(-2).swapaxes(-2, -1)
    output = inter_2 @ A2.squeeze(-2)
    Y2 = th.sum(output, dim=0)
    Y2 = Y2.reshape((7, 5, K, 768)).permute(2, 0, 1, 3)
    print(f"3D output Y shape: {Y2.shape}")
    return Y2


def mul_4D(w_, input_):
    K, E, H, D = w_.shape
    x_res = input_.reshape((7, 5, H, D))
    decomp = tl.decomposition.CP(rank=4, normalize_factors=False,
        verbose=False, init="random", tol=1e-24, random_state=42
    )
    _ , (A1, A2, A3, A4) = decomp.fit_transform(w_)
    recons_weight = tl.cp_to_tensor((None, (A1, A2, A3, A4)))
    print(th.allclose(recons_weight, w_))
    # A1 = preprocess(A1)
    # A2 = preprocess(A2)
    # A3 = preprocess(A3)
    # A4 = preprocess(A4)
    A1, A2, A3, A4 = map(preprocess, (A1, A2, A3, A4))
    inter_1 = x_res @ A4.swapaxes(-2, -1)
    inter_2 = A3 @ inter_1
    inter_3 = inter_2 @ A2
    Y3 = A1.swapaxes(-2, -1) @ inter_3
    Y3 = th.sum(Y3, 0).permute((2, 0, 1, 3))
    print(f"4D output Y shape: {Y3.shape}")
    return Y3


def mul_4DE(w_, input_):
    K, E, H, D = w_.shape
    x_res = input_.reshape((7, 5, H, D))
    decomp = tl.decomposition.CP(rank=4, normalize_factors=False,
        verbose=False, init="random", tol=1e-24, random_state=42
    )
    _ , (A1, A2, A3, A4) = decomp.fit_transform(w_)
    recons_weight = tl.cp_to_tensor((None, (A1, A2, A3, A4)))
    print(th.allclose(recons_weight, w_))
    A1 = preprocess(A1)
    A2 = preprocess(A2)
    A3 = preprocess(A3)
    A4 = preprocess(A4)
    i1 = th.einsum("...ij, ...jk -> ...ik", x_res, A4.swapaxes(-2, -1))
    i2 = th.einsum("...ij, ...jk -> ...ik", A3, i1)
    i3 = th.einsum("...ij, ...jk -> ...ik", i2, A2)
    Y4 = th.einsum("...ij, ...jk -> ...ik", A1.swapaxes(-2, -1), i3)
    Y4 = th.sum(Y4, 0).permute((2, 0, 1, 3))
    print(f"4D EINSUM output Y shape: {Y4.shape}")
    return Y4


def main():
    print (th.__version__)
    print (th.version.cuda)
    print (th.cuda.get_device_name())
    print (th.cuda.get_device_properties ('cuda').total_memory)

    weight = th.randn((3, 768, 12, 768//12)).to("cuda")
    X = th.randn((7, 5, 768)).to("cuda")
    global preprocess
    preprocess = lambda  x: x.unsqueeze(0).unsqueeze(0).unsqueeze(0).permute((-1, 0, 1, 2, 3))

    # Warmup GPU
    dummy = th.randn(768, 768*3).to("cuda")
    for _ in range(20):
        wt = th.matmul(X, dummy)
        wt = None
        wt = th.einsum("ijk, kl -> kl", X, dummy)
        wt = None
    
    # 2D Implementation
    th.cuda.reset_peak_memory_stats("cuda")
    th.cuda.synchronize()
    start = time.time()
    y_2d = mul_2D(weight, X)
    th.cuda.synchronize()
    end = time.time()
    print(f"2D time: {end-start}")
    print(f"2D memory (GB): {th.cuda.max_memory_allocated('cuda')/10**9}")

    # 3D Implementation
    th.cuda.reset_peak_memory_stats("cuda")
    th.cuda.synchronize()
    start = time.time()
    y_3d = mul_3D(weight, X)
    th.cuda.synchronize()
    end = time.time()
    print(f"3D time: {end-start}")
    print(f"3D memory (GB): {th.cuda.max_memory_allocated('cuda')/10**9}")

    # 4D Implementation
    th.cuda.reset_peak_memory_stats("cuda")
    th.cuda.synchronize()
    start = time.time()
    y_4d = mul_4D(weight, X)
    th.cuda.synchronize()
    end = time.time()
    print(f"4D time: {end-start}")
    print(f"4D memory (GB): {th.cuda.max_memory_allocated('cuda')/10**9}")

    # 4D-EINSUM Implementation
    th.cuda.reset_peak_memory_stats("cuda")
    th.cuda.synchronize()
    start = time.time()
    y_4de = mul_4DE(weight, X)
    th.cuda.synchronize()
    end = time.time()
    print(f"4D-EINSUM time: {end-start}")
    print(f"4D-EINSUM memory (GB): {th.cuda.max_memory_allocated('cuda')/10**9}")

    print(th.allclose(y_2d, y_3d), th.allclose(y_2d, y_4d), th.allclose(y_2d, y_4de))
    assert th.allclose(y_2d, y_3d)
    assert th.allclose(y_2d, y_4d)
    assert th.allclose(y_2d, y_4de)


if __name__ == "__main__":
    main()

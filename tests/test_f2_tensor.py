import pytest
import numpy as np
from qec_loss.f2_tensor import F2Tensor, PackedF2Matrix


def test_f2_tensor_init():
    # Test initialization with various dimensions
    t1 = F2Tensor([5])
    assert t1.shape == [5]
    assert t1.strides == [1]
    assert t1.offset == 0

    t2 = F2Tensor([3, 4])
    assert t2.shape == [3, 4]
    assert t2.strides == [4, 1]
    assert t2.offset == 0

    t3 = F2Tensor([2, 3, 4])
    assert t3.shape == [2, 3, 4]
    assert t3.strides == [12, 4, 1]
    assert t3.offset == 0

    # Test zero initialization
    arr1 = t1.to_numpy()
    assert np.all(arr1 == 0)
    arr2 = t2.to_numpy()
    assert np.all(arr2 == 0)


def test_f2_tensor_element_access():
    t = F2Tensor([3, 4])

    # Write values
    t[0, 0] = 1
    t[1, 2] = 1
    t[2, 3] = 1

    # Read values
    assert t[0, 0] == 1
    assert t[0, 1] == 0
    assert t[1, 2] == 1
    assert t[2, 3] == 1

    # Negative indexing
    assert t[-3, -4] == 1
    assert t[-2, -2] == 1
    assert t[-1, -1] == 1

    t[-1, -1] = 0
    assert t[2, 3] == 0

    # Index bounds checking
    with pytest.raises(IndexError):
        _ = t[3, 0]
    with pytest.raises(IndexError):
        _ = t[0, 4]
    with pytest.raises(IndexError):
        _ = t[-4, 0]
    with pytest.raises(IndexError):
        _ = t[0, -5]

    # Dimensionality mismatch
    with pytest.raises(ValueError):
        _ = t[0]
    with pytest.raises(ValueError):
        _ = t[0, 0, 0]


def test_f2_tensor_slicing_and_subviews():
    t = F2Tensor([4, 6])
    # Fill with some pattern:
    # 0 1 0 1 0 1
    # 1 0 1 0 1 0
    # 0 1 0 1 0 1
    # 1 0 1 0 1 0
    for i in range(4):
        for j in range(6):
            t[i, j] = (i + j) % 2

    # Subview slicing
    sub = t[1:3, 2:5]
    assert sub.shape == [2, 3]
    assert sub.strides == [6, 1]
    assert sub.offset == 1 * 6 + 2

    # Check values in subview
    assert sub[0, 0] == t[1, 2]
    assert sub[0, 1] == t[1, 3]
    assert sub[1, 2] == t[2, 4]

    # Zero-copy memory sharing verification:
    # Modify subview, verify changes in original
    sub[0, 0] = 0 if sub[0, 0] == 1 else 1
    assert t[1, 2] == sub[0, 0]

    t[2, 4] = 0 if t[2, 4] == 1 else 1
    assert sub[1, 2] == t[2, 4]

    # Setting a slice with a scalar
    sub[:, :] = 1
    assert np.all(sub.to_numpy() == 1)
    # Check that original tensor was modified in the sliced area
    assert np.all(t.to_numpy()[1:3, 2:5] == 1)

    # Setting a slice with another F2Tensor
    val_tensor = F2Tensor([2, 3])
    val_tensor[0, 0] = 1
    val_tensor[1, 2] = 1
    sub[:, :] = val_tensor
    assert t[1, 2] == 1
    assert t[2, 4] == 1
    assert t[1, 3] == 0


def test_f2_tensor_transpose():
    t = F2Tensor([3, 5])
    t[1, 2] = 1
    t[2, 4] = 1

    transposed = t.T
    assert transposed.shape == [5, 3]
    assert transposed.strides == [1, 5]
    assert transposed.offset == 0

    assert transposed[2, 1] == 1
    assert transposed[4, 2] == 1
    assert transposed[0, 0] == 0

    # Memory is shared
    transposed[0, 1] = 1
    assert t[1, 0] == 1


def test_f2_tensor_matmul():
    # Matrix multiplication over F_2 (AND & XOR)
    # A (3x2), B (2x4) -> C (3x4)
    A = F2Tensor([3, 2])
    B = F2Tensor([2, 4])

    # A:
    # 1 0
    # 0 1
    # 1 1
    A[0, 0] = 1
    A[1, 1] = 1
    A[2, 0] = 1
    A[2, 1] = 1

    # B:
    # 1 1 0 1
    # 0 1 1 1
    B[0, 0] = 1
    B[0, 1] = 1
    B[0, 3] = 1
    B[1, 1] = 1
    B[1, 2] = 1
    B[1, 3] = 1

    # Result over GF(2)
    # C[0, :] = [1, 1, 0, 1]
    # C[1, :] = [0, 1, 1, 1]
    # C[2, :] = [1, 1^1, 1, 1^1] = [1, 0, 1, 0]
    C = A @ B
    assert C.shape == [3, 4]
    assert C[0, 0] == 1 and C[0, 1] == 1 and C[0, 2] == 0 and C[0, 3] == 1
    assert C[1, 0] == 0 and C[1, 1] == 1 and C[1, 2] == 1 and C[1, 3] == 1
    assert C[2, 0] == 1 and C[2, 1] == 0 and C[2, 2] == 1 and C[2, 3] == 0

    # Using @ operator
    C_matmul = A @ B
    assert np.all(C_matmul.to_numpy() == C.to_numpy())

    # Dimension mismatch checks
    D = F2Tensor([3])
    with pytest.raises(ValueError):
        _ = A @ D

    E = F2Tensor([3, 3])
    with pytest.raises(ValueError):
        _ = A @ E


def test_f2_tensor_numpy_interop():
    t = F2Tensor([2, 3])
    t[0, 1] = 1
    t[1, 2] = 1

    # Using np.asarray should not copy
    arr = np.asarray(t)
    assert arr.shape == (2, 3)
    assert arr[0, 1] == 1
    assert arr[1, 2] == 1

    # Mutate numpy array, check F2Tensor
    arr[0, 0] = 1
    assert t[0, 0] == 1

    # Mutate F2Tensor, check numpy array
    t[1, 1] = 1
    assert arr[1, 1] == 1


def test_f2_tensor_str_repr():
    t = F2Tensor([2, 2])
    t[0, 0] = 1
    t[1, 1] = 1

    s = str(t)
    r = repr(t)

    assert "[[1 0]\n [0 1]]" in s
    assert "F2Tensor" in r


def test_f2_tensor_step_slicing():
    t = F2Tensor([6])
    for i in range(6):
        t[i] = i % 2

    # Slice with step 2
    sub = t[::2]
    assert sub.shape == [3]
    assert sub.strides == [2]
    assert sub[0] == t[0]
    assert sub[1] == t[2]
    assert sub[2] == t[4]

    # Mutate through step slice
    sub[1] = 1
    assert t[2] == 1


def test_f2_tensor_buffer_init():
    # Construct F2Tensor directly from a numpy array (zero-copy)
    arr = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    t = F2Tensor(arr)

    assert t.shape == [2, 3]
    assert t[0, 0] == 1
    assert t[0, 1] == 0
    assert t[0, 2] == 1
    assert t[1, 1] == 1

    # Mutate numpy array, verify C++ tensor changes (zero-copy)
    arr[0, 1] = 1
    assert t[0, 1] == 1

    # Mutate C++ tensor, verify numpy array changes (zero-copy)
    t[1, 0] = 1
    assert arr[1, 0] == 1


def test_f2_tensor_linear_algebra():
    # System A matrix (3x3):
    # 1 1 0
    # 1 0 1
    # 0 1 1
    A = F2Tensor(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8))

    # RREF and rank
    A_rref = A.rref()
    assert A_rref.shape == [3, 3]
    assert A.rank() == 2  # over GF(2), row 0 + row 1 = row 2, so rank is 2

    # Kernel (null space) basis
    # The kernel of A over GF(2) is span([1, 1, 1]) since:
    # [1 1 0]*[1 1 1]^T = 1^1 = 0
    # [1 0 1]*[1 1 1]^T = 1^1 = 0
    # [0 1 1]*[1 1 1]^T = 1^1 = 0
    K = A.kernel()
    assert K.shape == [1, 3]
    assert K[0, 0] == 1 and K[0, 1] == 1 and K[0, 2] == 1

    # Solve system Ax = b
    # Let b = [1, 0, 1]^T
    b = F2Tensor(np.array([1, 0, 1], dtype=np.uint8))
    x = A.solve(b)
    # Check that x is a solution: Ax = b
    # A * x:
    res = A @ F2Tensor(np.array([[x[0]], [x[1]], [x[2]]], dtype=np.uint8))
    assert res[0, 0] == b[0]
    assert res[1, 0] == b[1]
    assert res[2, 0] == b[2]

    # Test system with no solution (inconsistent)
    # Let b = [1, 1, 1]^T
    # Row 0 + Row 1 = [0, 1, 1]^T = Row 2
    # but b_0 + b_1 = 1^1 = 0 != b_2 = 1. Inconsistent!
    b_inconsistent = F2Tensor(np.array([1, 1, 1], dtype=np.uint8))
    with pytest.raises(RuntimeError):
        _ = A.solve(b_inconsistent)


def test_packed_f2_matrix():
    t = PackedF2Matrix(3, 4)
    assert t.rows == 3
    assert t.cols == 4

    # Test get/set with bracket indexing
    t[0, 0] = 1
    t[1, 2] = 1
    t[2, 3] = 1
    assert t[0, 0] == 1
    assert t[0, 1] == 0
    assert t[1, 2] == 1
    assert t[2, 3] == 1

    # Test transpose (.T)
    trans = t.T
    assert trans.rows == 4 and trans.cols == 3
    assert trans[0, 0] == 1
    assert trans[2, 1] == 1
    assert trans[3, 2] == 1
    assert trans[0, 1] == 0

    # Test multiplication
    # A (3x2), B (2x4) -> C (3x4)
    A = PackedF2Matrix(3, 2)
    B = PackedF2Matrix(2, 4)

    A[0, 0] = 1
    A[1, 1] = 1
    A[2, 0] = 1
    A[2, 1] = 1

    B[0, 0] = 1
    B[0, 1] = 1
    B[0, 3] = 1
    B[1, 1] = 1
    B[1, 2] = 1
    B[1, 3] = 1

    C = A @ B
    assert C.rows == 3 and C.cols == 4
    assert C[0, 0] == 1 and C[0, 1] == 1 and C[0, 2] == 0 and C[0, 3] == 1
    assert C[1, 0] == 0 and C[1, 1] == 1 and C[1, 2] == 1 and C[1, 3] == 1
    assert C[2, 0] == 1 and C[2, 1] == 0 and C[2, 2] == 1 and C[2, 3] == 0

    # Test matrix multiplication operator @
    C_matmul = A @ B
    for i in range(3):
        for j in range(4):
            assert C_matmul[i, j] == C[i, j]

    # Test Linear Algebra: RREF and rank
    # System matrix (3x3):
    # 1 1 0
    # 1 0 1
    # 0 1 1
    sys_mat = PackedF2Matrix(3, 3)
    sys_mat[0, 0] = 1
    sys_mat[0, 1] = 1
    sys_mat[0, 2] = 0
    sys_mat[1, 0] = 1
    sys_mat[1, 1] = 0
    sys_mat[1, 2] = 1
    sys_mat[2, 0] = 0
    sys_mat[2, 1] = 1
    sys_mat[2, 2] = 1

    rref_mat = sys_mat.rref()
    assert rref_mat.rows == 3 and rref_mat.cols == 3
    assert sys_mat.rank() == 2

    # Test kernel
    K = sys_mat.kernel()
    assert K.rows == 1 and K.cols == 3
    assert K[0, 0] == 1 and K[0, 1] == 1 and K[0, 2] == 1

    # Test solve
    # Let b = [1, 0, 1]^T
    b = PackedF2Matrix(3, 1)
    b[0, 0] = 1
    b[1, 0] = 0
    b[2, 0] = 1

    x = sys_mat.solve(b)
    assert x.rows == 3 and x.cols == 1

    # Check that A @ x = b
    res = sys_mat @ x
    assert res[0, 0] == b[0, 0]
    assert res[1, 0] == b[1, 0]
    assert res[2, 0] == b[2, 0]

    # Test direct XOR helpers (xor_bit and xor_rows)
    # 1. xor_bit: Toggle a bit
    t.xor_bit(0, 0)
    assert t[0, 0] == 0  # originally 1, now toggled to 0
    t.xor_bit(0, 0)
    assert t[0, 0] == 1  # toggled back to 1

    # 2. xor_rows: XOR row 1 into row 0
    # t originally:
    # 1 0 0 0
    # 0 0 1 0
    # 0 0 0 1
    # After xor_rows(0, 1): row 0 becomes row 0 ^ row 1 = [1, 0, 1, 0]
    t.xor_rows(0, 1)
    assert t[0, 0] == 1 and t[0, 1] == 0 and t[0, 2] == 1 and t[0, 3] == 0
    assert t[1, 0] == 0 and t[1, 1] == 0 and t[1, 2] == 1 and t[1, 3] == 0

    # 3. one and zero: Force bit state
    t.one(1, 1)
    assert t[1, 1] == 1
    t.zero(1, 1)
    assert t[1, 1] == 0
    t.one(1, 1)
    assert t[1, 1] == 1
    t.zero(0, 0)
    assert t[0, 0] == 0

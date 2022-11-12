from jax.lax.linalg import svd
from typing import Union, Tuple
from jax.numpy import ndarray
from jax._src.typing import Array

_engine_list = ["jax", "rust", "lapack", "randomized"]


def _svd(
    matrix: Union[Array, ndarray],
    full_matrices: bool = False,
    compute_uv: bool = True,
    engine: str = "jax",
) -> Union[Array, Tuple[Array, Array, Array]]:
    if engine == "jax":
        matrices = svd(
            matrix, full_matrices=full_matrices, compute_uv=compute_uv
        )  # return either the U,V^t matrices and the s vector or just the s vector
    elif engine == "rust":
        NotImplementedError("engine support for rust is not yet implemented.")
    else:
        ValueError(f'engine should be one of {" ".join(_engine_list)}')
    return matrices

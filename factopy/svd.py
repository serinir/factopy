from jax.lax.linalg import svd
from typing import Union, Tuple
import jax.numpy as jnp
from jax._src.typing import Array

_engine_list = ["jax", "rust", "lapack", "randomized"]


def _svd(
    matrix: Union[Array, jnp.ndarray],
    full_matrices: bool = False,
    compute_uv: bool = True,
    engine: str = "jax",
) -> Union[Array, Tuple[Array, Array, Array]]:
    if engine == "jax":
        matrices = svd(
            matrix, full_matrices=full_matrices, compute_uv=compute_uv
        )  # return either the U,V^t matrices and the s vector or just the s vector
    elif engine == "rust":
        raise NotImplementedError("engine support for rust is not yet implemented.")
    else:
        raise ValueError(f'engine should be one of {" ".join(_engine_list)}')
    return matrices


def _svd_flip(u, v):
    max_abs_cols = jnp.argmax(jnp.abs(u), axis=0)
    signs = jnp.sign(u[max_abs_cols, jnp.array(range(u.shape[1]))])
    u *= signs
    v *= signs[:, jnp.newaxis]
    return u, v

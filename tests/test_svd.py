from jax.lax.linalg import svd
from jax.numpy.linalg import svd as jnsvd
from numpy.linalg import svd as nsvd
from sklearn.utils.extmath import randomized_svd

import jax.numpy as jnp
from jax import random
from factopy.svd import _svd

thershold = 1e-4


class TestSVD:
    A = random.uniform(random.PRNGKey(42), shape=(100, 200), dtype=float)

    U, s, V = _svd(A)
    Uc, sc, Vc = _svd(A, full_matrices=True)

    def test_lax(self):
        U2, s2, V2 = svd(self.A, full_matrices=False)
        assert self.U.shape == U2.shape
        assert self.s.shape == s2.shape
        assert self.V.shape == V2.shape
        assert jnp.allclose(self.U, U2, thershold)
        assert jnp.allclose(self.s, s2, thershold)
        assert jnp.allclose(self.V, V2, thershold)

    def test_lax_complete(self):
        U2, s2, V2 = svd(self.A)
        assert jnp.allclose(self.Uc, U2, thershold)
        assert jnp.allclose(self.sc, s2, thershold)
        assert jnp.allclose(self.Vc, V2, thershold).sum()

    def test_jnp(self):
        U2, s2, V2 = jnsvd(self.A, full_matrices=False)
        assert self.U.shape == U2.shape
        assert self.s.shape == s2.shape
        assert self.V.shape == V2.shape
        assert jnp.allclose(self.U, U2, thershold)
        assert jnp.allclose(self.s, s2, thershold)
        assert jnp.allclose(self.V, V2, thershold)

    def test_jnp_complete(self):
        U2, s2, V2 = jnsvd(self.A)

        assert jnp.allclose(self.Uc, U2, thershold)
        assert jnp.allclose(self.sc, s2, thershold)
        assert jnp.allclose(self.Vc, V2, thershold)

    def test_numpy(self):
        U2, s2, V2 = nsvd(self.A, full_matrices=False)
        assert self.U.shape == U2.shape
        assert self.s.shape == s2.shape
        assert self.V.shape == V2.shape
        # assert jnp.allclose(self.U,U2,thershold)
        assert jnp.allclose(self.s, s2, thershold)
        # assert jnp.allclose(self.V,V2,thershold)

    def test_numpy_complete(self):
        U2, s2, V2 = nsvd(self.A)
        assert self.Uc.shape == U2.shape
        assert self.sc.shape == s2.shape
        assert self.Vc.shape == V2.shape
        # assert jnp.allclose(self.Uc,U2,thershold)
        assert jnp.allclose(self.sc, s2, thershold)
        # assert jnp.allclose(self.Vc,V2,thershold)

    def test_randomized_svd(self):
        U2, s2, V2 = randomized_svd(
            self.A, n_components=self.A.shape[0], random_state=None
        )
        assert self.U.shape == U2.shape
        assert self.s.shape == s2.shape
        assert self.V.shape == V2.shape
        # assert jnp.allclose(self.U,U2,thershold)
        assert jnp.allclose(self.s, s2, thershold)
        # assert jnp.allclose(self.V,V2,thershold)

import jax.numpy as jnp
from factopy.pca import PCA
from jax import random
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SKPCA
import pytest


class TestPCA:
    model = PCA(n_component=2)
    skmodel = SKPCA(n_components=2)
    # X = random.uniform(random.PRNGKey(42), shape=(10, 10), dtype=float)
    X = load_iris().data
    X_mean_ = jnp.mean(X, axis=0)
    centered_X = X - X_mean_
    total_inertia_ = jnp.sum(jnp.square(X)) / len(X)

    def test_explained_variance_fullmatrice(self):
        self.skmodel.fit(self.X)
        self.model.fit(self.X)

        assert jnp.allclose(
            self.model.explained_variance_, self.skmodel.explained_variance_
        )

    def test_components(self):
        self.skmodel.fit(self.X)
        self.model.fit(self.X)
        print(self.model.components_.flatten())
        print(self.skmodel.components_.flatten())
        thershold = 5e-2
        assert jnp.allclose(
            self.model.components_, self.skmodel.components_, rtol=thershold
        )

    def test_raised_error(self):
        with pytest.raises(ValueError):
            model = PCA(n_component=None, svd_solver="other")
            model.fit(self.X)
    @pytest.mark.parametrize("ncomp", list(range(1,min(X.shape[0],X.shape[1]))))
    def test_results(self,ncomp):
        self.model = PCA(n_component=ncomp)
        self.skmodel = SKPCA(n_components=ncomp)
        results_pca_other = self.skmodel.fit_transform(self.X)
        results_pca = self.model.fit_transform(self.X)
        thershold = 5e-2
        print(results_pca)
        print(results_pca_other)
        assert jnp.allclose(results_pca, results_pca_other,thershold)

import jax.numpy as jnp
from factopy.pca import PCA
from jax import random
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SKPCA
class TestPCA:
    model = PCA(n_component=None)
    skmodel = SKPCA()
    # X = random.uniform(random.PRNGKey(42), shape=(10,10), dtype=float)
    X = load_iris().data
    X_mean_ = jnp.mean(X,axis=0)
    centered_X = X - X_mean_
    total_inertia_ = jnp.sum(jnp.square(X)) / len(X)

    def test_explained_variance_fullmatrice(self):
        self.skmodel.fit(self.X)
        self.model.fit(self.X)
        
        assert jnp.allclose(
            self.model.explained_variance_,
            self.skmodel.explained_variance_
        )
    def test_components(self):
        self.skmodel.fit(self.X)
        self.model.fit(self.X)
        # print(self.model.components_.flatten())
        # print(self.skmodel.components_.flatten())\
        thershold = 5e-2
        for x,y in (zip(list(map(float,self.model.components_.flatten())), list(self.skmodel.components_.flatten()))):
            test = jnp.isclose(
                abs(x),
                abs(y),
                rtol=thershold
                )
            if not test :
                print(x,y)   
        assert jnp.allclose(
            self.model.components_,
            self.skmodel.components_,
            rtol=thershold
        )
import jax.numpy as jnp
from factopy.pca import PCA
from jax import random
from sklearn.decomposition import PCA as SKPCA
class TestPCA:
    def test_covariance(self):
        model = PCA()
        skmodel = SKPCA()
        A = random.uniform(random.PRNGKey(42), shape=(5, 5), dtype=float)
       
        results = skmodel.fit_transform(A)

        U,s,V = model._fit ((A- A.mean())/A.std())

        print(skmodel.explained_variance_)
        print((s**2)/4)

        assert False



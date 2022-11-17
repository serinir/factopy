import jax.numpy as jnp
import pandas as pd
from sklearn import base
from typing import Union
from jax._src.typing import Array
from pandas import DataFrame
from scipy.sparse import issparse
from factopy.svd import _svd, _svd_flip

# from sklearn import preprocessing
# from sklearn import utils


class PCA(base.BaseEstimator, base.TransformerMixin):
    """Principal Component Analysis (PCA).
    Parameters:
        n_component (int): number of dimensions kept in the result
        scale_unit (bool, optional): to choose whether to scale the datas or not. Defaults to True.
        quanti_sup (Union[Array,int,str], optional): vector of the indexes of the quantitative supplementary variables. Defaults to None.
        quali_sup (Union[Array,int,str], optional): vector of the indexes of the qualitative supplementary variables. Defaults to None.
        graph (bool, optional): to choose whether to plot the graphs or not.  Defaults to False.
        svd_solver (bool, optional) :{'auto', 'full', 'arpack','','randomized'}
            auto -> either randomized or qdwh.
            lapack -> LAPACK.
            arpack -> ARPACK through sklearn trauncated svd.
            qdwh -> qdwh-svd through jax.
            randomized -> sklear. randomized svd Halko et al method.

    """

    def __init__(
        self,
        n_component: Union[int, str],
        scale_unit: bool = True,
        quanti_sup: Union[Array, int, str] = None,
        quali_sup: Union[Array, int, str] = None,
        svd_solver: str = "qdwh",
        graph: bool = False,
    ) -> None:

        super().__init__()
        self.n_component = n_component
        self.scale_unit = scale_unit
        self.quanti_sup = quanti_sup
        self.quali_sup = quali_sup
        self.svd_solver = svd_solver
        self.graph = graph

    def fit(self, X: Union[DataFrame, Array], y=None):
        if issparse(X):
            Warning("X is sparced solver switched to trauncatedSVD")
        # when ncomp is None:
        fit_ncomponents = min(X.shape)
        # when svd solver is auto :
        fit_svd_solver = self.svd_solver
        if fit_svd_solver == "auto":
            if max(X.shape > 500):
                fit_svd_solver = "qdwh"
            elif fit_ncomponents < 0.8 * min(X.shape):
                fit_svd_solver = "randomized"

        elif fit_svd_solver == "qdwh":
            self._fit_qdwh(X, fit_ncomponents)
        # elif fit_svd_solver == "lapack":
        #     self._fit_lapack(X, fit_ncomponents)
        # elif fit_svd_solver in ["arpack", "randomized"]:
        #     self._fit_truncated(X, fit_ncomponents, fit_svd_solver)
        else:
            raise ValueError("Unrecognized svd_solver='{0}'".format(fit_svd_solver))

        return self

    def _fit_qdwh(self, X, ncomponent=None):

        self.mean_ = jnp.mean(X, axis=0)
        centered_X = X - self.mean_
        U, s, Vt = _svd(centered_X)
        U, Vt = _svd_flip(U, Vt)
        self.explained_variance_ = jnp.square(s) / (X.shape[0] - 1)
        self.total_explained_variance_ = jnp.sum(self.explained_variance_)
        self.explained_variance_ratio = (
            self.explained_variance_ / self.total_explained_variance_
        )

        self.components_ = Vt
        return U, s, Vt

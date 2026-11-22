"""Contains transformers that apply numeric functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import narwhals as nw
import numpy as np
from beartype import beartype
from sklearn.cluster import KMeans

from tubular._utils import (
    _convert_dataframe_to_narwhals,
    _return_narwhals_or_native_dataframe,
    block_from_json,
)
from tubular.base import BaseTransformer, register
from tubular.functions.numeric import (
    get_difference_of_two_columns,
    get_ratio_of_two_columns,
)
from tubular.mixins import (
    CheckNumericMixin,
)
from tubular.types import (
    DataFrame,
    FloatTypeAnnotated,
    ListOfOneStr,
    ListOfTwoStrs,
)

if TYPE_CHECKING:
    from narwhals.typing import FrameT, IntoSeriesT


@register
class BaseNumericTransformer(BaseTransformer, CheckNumericMixin):
    """Extends BaseTransformer for datetime scenarios.

    Attributes
    ----------
    columns : List[str]
        List of columns to be operated on

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Examples
    --------
    ```pycon
    >>> BaseNumericTransformer(
    ...     columns="a",
    ... )
    BaseNumericTransformer(columns=['a'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = False

    FITS = False

    def __init__(self, columns: list[str], **kwargs: dict[str, bool]) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : List[str]
            List of columns to be operated on.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, **kwargs)

    def fit(
        self,
        X: DataFrame,
        y: nw.Series | None = None,
    ) -> BaseNumericTransformer:
        """Validate data and attributes prior to the child objects fit logic.

        Parameters
        ----------
        X : DataFrame
            A dataframe containing the required columns

        y : Series | None
            Required for pipeline.

        Returns
        -------
            BaseNumericTransformer:
                fitted class instance.

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = BaseNumericTransformer(
        ...     columns="a",
        ... )

        >>> test_df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        >>> transformer.fit(test_df)
        BaseNumericTransformer(columns=['a'])

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        super().fit(X, y)

        CheckNumericMixin.check_numeric_columns(self, X.select(self.columns))

        return self

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: bool | None = None,
    ) -> DataFrame:
        """Validate data and attributes prior to the child objects transform logic.

        Parameters
        ----------
        X : DataFrame
            Data to transform.

        return_native_override: Optional[bool]
            Option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : DataFrame
            Validated data

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = BaseNumericTransformer(
        ...     columns="a",
        ... )

        >>> test_df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        >>> # base class has no effect on datag
        >>> transformer.transform(test_df)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        └─────┴─────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)
        return_native = self._process_return_native(return_native_override)
        X = super().transform(X, return_native_override=False)

        CheckNumericMixin.check_numeric_columns(self, X.select(self.columns))

        return _return_narwhals_or_native_dataframe(X, return_native)


def _assign_to_buckets(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each value to the bucket of its nearest centre.

    Bucket boundaries are the midpoints between consecutive sorted
    centres. A value sitting exactly on a boundary is assigned to the
    lower bucket, i.e. the bucket with the smaller centre. The rule is
    deterministic so that assignments can be reviewed and reproduced
    outside of this package.

    Parameters
    ----------
    values : np.ndarray
        Values to assign to buckets.

    centers : np.ndarray
        Bucket centres, sorted ascending with no duplicates.

    Returns
    -------
    np.ndarray
        Integer bucket labels, aligned with the order of centers.

    """
    boundaries = centers[:-1] + (centers[1:] - centers[:-1]) / 2

    return np.digitize(values, boundaries, right=True)


@register
class OneDKmeansTransformer(BaseNumericTransformer):
    """Discretises a numeric column into buckets found by 1D k-means.

    Transformer runs the kmeans algorithm on the column and treats the
    cluster centres as bucket centres. If ``min_explained_dispersion`` is
    set, fit selects the smallest number of buckets that explains at least
    that share of the total dispersion (sum of squared distances to the
    column mean), never exceeding ``n_clusters`` buckets. Otherwise
    exactly ``n_clusters`` buckets are formed.

    Fit stores the selected centres (``centers_``), the effective
    population of each bucket in the fit data (``bucket_population_``) and
    the share of dispersion explained by the selected buckets
    (``explained_dispersion_``). Transform only uses ``centers_``: each
    value is assigned to the bucket of its nearest centre.

    Attributes
    ----------
    centers_ : list of float
        Bucket centres selected at fit, sorted ascending. This is the only
        fitted attribute used by transform.

    bucket_population_ : list of int
        Effective population of each bucket in the fit data, aligned with
        ``centers_``. Stored for review and monitoring, not used by
        transform.

    explained_dispersion_ : float
        Share of the total dispersion explained by the selected buckets.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Notes
    -----
    Bucket assignment follows deterministic, reviewable rules:

    - the number of buckets never exceeds the number of distinct values
      seen at fit time (empty or duplicated clusters are not created);
    - identical centres are merged into a single bucket at fit time;
    - a value exactly halfway between two centres is assigned to the
      lower bucket (the one with the smaller centre);
    - equal values are always assigned to the same bucket, so duplicate
      values each count towards ``bucket_population_``.

    Examples
    --------
    ```pycon
    >>> OneDKmeansTransformer(
    ...     columns="a",
    ...     n_clusters=2,
    ...     new_column_name="new",
    ...     kmeans_kwargs={"random_state": 42},
    ... )
    OneDKmeansTransformer(columns=['a'], kmeans_kwargs={'random_state': 42},
                          n_clusters=2, new_column_name='new')

    ```

    """

    polars_compatible = True

    lazyframe_compatible = False

    jsonable = True

    FITS = True

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Serialize the transformer to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, dict[str, Any]]:
            JSON representation of the transformer, including init parameters.

        Examples
        --------
        >>> import polars as pl
        >>> x = OneDKmeansTransformer(
        ... columns='a',
        ... n_clusters=2,
        ... new_column_name="new",
        ... kmeans_kwargs={"random_state": 42},
        ...    )
        >>> test_df=pl.DataFrame({'a': [1,2,3,4],  'b': [5,6,7,8]})
        >>> x.fit(test_df)
        OneDKmeansTransformer(columns=['a'], kmeans_kwargs={'random_state': 42},
                              n_clusters=2, new_column_name='new')
        >>> x.to_json()
        {'tubular_version': ..., 'classname': 'OneDKmeansTransformer', 'init': {'columns': ['a'], 'copy': False, 'verbose': False, 'return_native': True, 'new_column_name': 'new', 'n_init': 'auto', 'n_clusters': 2, 'kmeans_kwargs': {'random_state': 42}, 'min_explained_dispersion': None}, 'fit': {'is_fitted_': True, 'centers_': [2.0, 4.0], 'bucket_population_': [3, 1], 'explained_dispersion_': 0.6}}

        """
        self.check_is_fitted(["centers_", "bucket_population_"])
        json_dict = super().to_json()

        json_dict["init"].update(
            {
                "new_column_name": self.new_column_name,
                "n_init": self.n_init,
                "n_clusters": self.n_clusters,
                "kmeans_kwargs": self.kmeans_kwargs,
                "min_explained_dispersion": self.min_explained_dispersion,
            },
        )
        json_dict["fit"]["centers_"] = self.centers_
        json_dict["fit"]["bucket_population_"] = self.bucket_population_
        json_dict["fit"]["explained_dispersion_"] = self.explained_dispersion_

        return json_dict

    @beartype
    def __init__(  # noqa: PLR0917, PLR0913
        self,
        columns: str | ListOfOneStr,
        new_column_name: str,
        n_init: str | int = "auto",
        n_clusters: int = 8,
        kmeans_kwargs: dict[str, object] | None = None,
        min_explained_dispersion: float | None = None,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : str or list[str]
            Name of the column to discretise.

        new_column_name : str
            Name given to the new discrete column.

        n_clusters : int, default = 8
            The number of clusters to form as well as the number of centroids to generate.
            When min_explained_dispersion is set, this acts as the upper cap on the
            number of buckets selected at fit.

        n_init: "auto" or int, default="auto"
            Number of times the k-means algorithm is run with different centroid seeds.
            The final results is the best output of n_init consecutive runs in terms of inertia.
            Several runs are recommended for sparse high-dimensional problems (see `Clustering sparse data with k-means <https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#kmeans-sparse-high-dim>`__).

            When n_init='auto', the number of runs depends on the value of init: 10 if using init='random' or init is a callable;
            1 if using init='k-means++' or init is an array-like.(Init is an arg in kmeans_kwargs. If init is not set then it defaults to k-means++ so n_init defaults to 1)

        kmeans_kwargs : dict, default = {}
            A dictionary of keyword arguments to be passed to the sklearn KMeans method when it is called in fit.

        min_explained_dispersion : float or None, default = None
            Minimum share of the total dispersion (between 0 and 1) that the
            selected buckets must explain. If None, exactly n_clusters buckets
            are formed. If set, fit tries increasing bucket counts from 1 up to
            n_clusters and keeps the smallest count reaching this threshold;
            n_clusters is used when no smaller count reaches it.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init().

        Raises
        ------
        ValueError:
            if n_clusters is below 1 or min_explained_dispersion is outside
            [0, 1].

        """
        if kmeans_kwargs is None:
            kmeans_kwargs = {}

        if n_clusters < 1:
            msg = (
                f"{self.classname()}: n_clusters must be at least 1, got {n_clusters}."
            )
            raise ValueError(msg)

        if min_explained_dispersion is not None and not (
            0.0 <= min_explained_dispersion <= 1.0
        ):
            msg = (
                f"{self.classname()}: min_explained_dispersion must be between "
                f"0 and 1, got {min_explained_dispersion}."
            )
            raise ValueError(msg)

        self.n_clusters = n_clusters
        self.new_column_name = new_column_name
        self.n_init = n_init
        self.kmeans_kwargs = kmeans_kwargs
        self.min_explained_dispersion = min_explained_dispersion

        if isinstance(columns, str):
            self.columns = [columns]
        else:
            self.columns = columns

        super().__init__(columns=self.columns, **kwargs)

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer

        Examples
        --------
        ```pycon
        >>> transformer = OneDKmeansTransformer(
        ...     columns="a",
        ...     n_clusters=2,
        ...     new_column_name="kmeans_column",
        ...     kmeans_kwargs={"random_state": 42},
        ... )

        >>> transformer.get_feature_names_out()
        ['kmeans_column']

        ```

        """
        return [
            self.new_column_name,
        ]

    @block_from_json
    @nw.narwhalify
    def fit(self, X: FrameT, y: IntoSeriesT | None = None) -> OneDKmeansTransformer:
        """Fit transformer to input data.

        Learns the bucket centres and stores them in ``centers_`` together
        with the effective population of each bucket in the fit data
        (``bucket_population_``) and the share of dispersion explained by
        the selected buckets (``explained_dispersion_``). If
        ``min_explained_dispersion`` is set, the smallest bucket count
        reaching that threshold (capped at ``n_clusters``) is selected.

        Parameters
        ----------
        X : pd/pl.DataFrame
            Dataframe with columns to learn scaling values from.

        y : None
            Required for pipeline.

        Returns
        -------
            OneDKmeansTransformer:
                Fitted class instance.

        Raises
        ------
        ValueError:
            if columns in X contain missing values.

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = OneDKmeansTransformer(
        ...     columns="a",
        ...     n_clusters=2,
        ...     new_column_name="new",
        ...     kmeans_kwargs={"random_state": 42},
        ... )

        >>> test_df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})

        >>> transformer.fit(test_df)
        OneDKmeansTransformer(columns=['a'], kmeans_kwargs={'random_state': 42},
                              n_clusters=2, new_column_name='new')

        ```

        """
        super().fit(X, y)

        X = nw.from_native(X)

        # Check that X does not contain Nans and return ValueError.
        if (
            X.select(nw.col(self.columns[0]).is_null().any()).to_numpy().ravel()[0]
            or X.select(nw.col(self.columns[0]).is_nan().any()).to_numpy().ravel()[0]
        ):
            msg = f"{self.classname()}: X should not contain missing values."
            raise ValueError(msg)

        values = X.select(self.columns[0]).to_numpy().ravel().astype(np.float64)

        # reviewable rule: buckets are capped at the number of distinct
        # values, extra clusters would be empty or duplicated
        n_unique = np.unique(values).size
        max_buckets = min(self.n_clusters, n_unique)

        total_dispersion = float(np.sum((values - values.mean()) ** 2))

        if self.min_explained_dispersion is None:
            candidate_bucket_counts = [max_buckets]
        else:
            candidate_bucket_counts = range(1, max_buckets + 1)

        best_model = None
        explained_dispersion = 0.0
        for n_buckets in candidate_bucket_counts:
            kmeans = KMeans(
                n_clusters=n_buckets,
                n_init=self.n_init,
                **self.kmeans_kwargs,
            )
            best_model = kmeans.fit(values.reshape(-1, 1))

            if n_unique == 1:
                # a single distinct value means there is no dispersion to
                # explain, one bucket trivially explains everything
                explained_dispersion = 1.0
            else:
                # clamp away float noise, the explained share lies in [0, 1]
                explained_dispersion = min(
                    1.0,
                    max(0.0, 1.0 - best_model.inertia_ / total_dispersion),
                )

            if (
                self.min_explained_dispersion is not None
                and explained_dispersion >= self.min_explained_dispersion
            ):
                # smallest bucket count explaining enough of the dispersion
                break

        # reviewable rule: identical centres are merged into a single
        # bucket, np.unique also sorts the centres ascending
        centers = np.unique(best_model.cluster_centers_.ravel())

        assignments = _assign_to_buckets(values, centers)

        self.centers_ = centers.tolist()
        self.bucket_population_ = np.bincount(
            assignments,
            minlength=centers.size,
        ).tolist()
        self.explained_dispersion_ = float(explained_dispersion)
        self.is_fitted_ = True
        return self

    @nw.narwhalify
    def transform(self, X: FrameT) -> FrameT:
        """Assign each row to the bucket of its nearest fitted centre.

        Only the centres stored at fit are used; the bucket populations
        are fit-time review information and play no part in scoring.

        Parameters
        ----------
        X : pl/pd.DataFrame
            Data to transform.

        Returns
        -------
        X : pl/pd.DataFrame
            Input X with additional cluster column added.

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> transformer = OneDKmeansTransformer(
        ...     columns="a",
        ...     n_clusters=2,
        ...     new_column_name="new",
        ...     kmeans_kwargs={"random_state": 42},
        ... )

        >>> test_df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})

        >>> _ = transformer.fit(test_df)
        >>> transformer.transform(test_df)
        shape: (4, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ new │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 5   ┆ 0   │
        │ 2   ┆ 6   ┆ 0   │
        │ 3   ┆ 7   ┆ 0   │
        │ 4   ┆ 8   ┆ 1   │
        └─────┴─────┴─────┘

        ```

        """
        X = super().transform(X)

        X = nw.from_native(X)
        native_backend = nw.get_native_namespace(X).__name__

        values = X.select(self.columns[0]).to_numpy().ravel().astype(np.float64)

        groups = _assign_to_buckets(
            values,
            np.asarray(self.centers_, dtype=np.float64),
        )

        return X.with_columns(
            nw.new_series(
                name=self.new_column_name,
                values=groups,
                backend=native_backend,
            ),
        )


@register
class DifferenceTransformer(BaseNumericTransformer):
    """Transformer that performs subtraction operation between two columns.

    This transformer allows performing subtraction between two columns in a DataFrame
    and stores the result in a new column.

    Attributes
    ----------
    columns : ListOfTwoStrs
        List of exactly two column names to operate on. The second column is subtracted from the first.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Examples
    --------
    ```pycon
    >>> transformer = DifferenceTransformer(columns=["a", "b"])
    >>> transformer.columns
    ['a', 'b']

    ```

    """

    polars_compatible = True

    FITS = False

    jsonable = True

    lazyframe_compatible = True

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs,
        **kwargs: bool | None,
    ) -> None:
        """Initialize the DifferenceTransformer.

        Parameters
        ----------
        columns : ListOfTwoStrs
            List of exactly two column names to operate on. The second column is subtracted from the first.
        verbose : bool, default=False
            Whether to print verbose output during transformation.
        kwargs: bool
            arguments for base class, e.g. verbose.

        """
        super().__init__(columns=columns, **kwargs)

        # Set new_column_name or generate a default one
        self.new_column_name = f"{columns[0]}_minus_{columns[1]}"
        self.is_fitted_ = True  # Does not fit

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return get_difference_of_two_columns(columns=self.columns)

    @beartype
    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Transform the DataFrame by applying the subtraction operation between two columns.

        Parameters
        ----------
        X : DataFrame
            DataFrame containing the columns to operate on.

        Returns
        -------
        DataFrame
            Transformed DataFrame with the new column containing the subtraction results.


        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> transformer = DifferenceTransformer(columns=["a", "b"])
        >>> test_df = pl.DataFrame({"a": [100, 200, 300], "b": [80, 150, 200]})
        >>> transformer.transform(test_df)
        shape: (3, 3)
        ┌─────┬─────┬───────────┐
        │ a   ┆ b   ┆ a_minus_b │
        │ --- ┆ --- ┆ ---       │
        │ i64 ┆ i64 ┆ i64       │
        ╞═════╪═════╪═══════════╡
        │ 100 ┆ 80  ┆ 20        │
        │ 200 ┆ 150 ┆ 50        │
        │ 300 ┆ 200 ┆ 100       │
        └─────┴─────┴───────────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, return_native_override=False)

        transform_expr = self.get_transform_exprs()

        X = X.with_columns(transform_expr)

        return _return_narwhals_or_native_dataframe(X, self.return_native)

    def get_feature_names_out(self) -> list[str]:
        """Get the names of the output features.

        Returns
        -------
        list[str]
            List containing the name of the new column created by the transformation.

        """
        return [f"{self.columns[0]}_minus_{self.columns[1]}"]


@register
class RatioTransformer(BaseNumericTransformer):
    """Transformer that performs division operation between two columns.

    This transformer allows performing division between two columns in a DataFrame
    and stores the result in a new column.

    Attributes
    ----------
    columns : ListOfTwoStrs
        List of exactly two column names to operate on. The first column is the numerator,
        and the second column is the denominator.
    return_dtype : str
        The dtype of the resulting column, either 'Float32' or 'Float64'.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Examples
    --------
    ```pycon
    >>> transformer = RatioTransformer(columns=["a", "b"], return_dtype="Float32")
    >>> transformer.columns
    ['a', 'b']
    >>> transformer.return_dtype
    'Float32'

    ```

    """

    polars_compatible = True

    FITS = False

    jsonable = True

    lazyframe_compatible = True

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Serialize the transformer to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, dict[str, Any]]:
            JSON representation of the transformer, including init parameters.

        Examples
        --------
        ```pycon
        >>> ratio_transformer = RatioTransformer(columns=["a", "b"], return_dtype="Float32")
        >>> ratio_transformer.to_json()
        {'tubular_version': ..., 'classname': 'RatioTransformer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'return_dtype': 'Float32'}, 'fit': {'is_fitted_': True}}

        ```

        """
        json_dict = super().to_json()
        json_dict["init"]["return_dtype"] = self.return_dtype

        return json_dict

    @beartype
    def __init__(
        self,
        columns: ListOfTwoStrs,
        return_dtype: FloatTypeAnnotated = "Float32",
        **kwargs: bool | None,
    ) -> None:
        """Initialize the RatioTransformer.

        Parameters
        ----------
        columns : ListOfTwoStrs
            List of exactly two column names to operate on. The first column is the numerator,
            and the second column is the denominator.
        return_dtype : str, default='Float32'
            The dtype of the resulting column, either 'Float32' or 'Float64'.
        kwargs: bool
            arguments for base class, e.g. verbose

        """
        super().__init__(columns=columns, **kwargs)

        self.return_dtype = return_dtype
        self.is_fitted_ = True  # Does not fit

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return get_ratio_of_two_columns(
            columns=self.columns,
            return_dtype=self.return_dtype,
        )

    @beartype
    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Transform the DataFrame by applying the division operation between two columns.

        Parameters
        ----------
        X : DataFrame
            DataFrame containing the columns to operate on.

        Returns
        -------
        DataFrame
            Transformed DataFrame with the new column containing the division results.

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> transformer = RatioTransformer(columns=["a", "b"], return_dtype="Float32")
        >>> test_df = pl.DataFrame({"a": [100, 200, 300], "b": [80, 150, 200]})
        >>> transformer.transform(test_df)
        shape: (3, 3)
        ┌─────┬─────┬────────────────┐
        │ a   ┆ b   ┆ a_divided_by_b │
        │ --- ┆ --- ┆ ---            │
        │ i64 ┆ i64 ┆ f32            │
        ╞═════╪═════╪════════════════╡
        │ 100 ┆ 80  ┆ 1.25           │
        │ 200 ┆ 150 ┆ 1.333333       │
        │ 300 ┆ 200 ┆ 1.5            │
        └─────┴─────┴────────────────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)
        X = super().transform(X, return_native_override=False)

        transform_expr = self.get_transform_exprs()

        X = X.with_columns(transform_expr)

        return _return_narwhals_or_native_dataframe(X, self.return_native)

    def get_feature_names_out(self) -> list[str]:
        """Get the names of the output features.

        Returns
        -------
        list[str]
            List containing the name of the new column created by the transformation.

        """
        return [f"{self.columns[0]}_divided_by_{self.columns[1]}"]

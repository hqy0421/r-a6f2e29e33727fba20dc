import pytest
from beartype.roar import BeartypeCallHintParamViolation

import tests.test_data as d
import tests.utils as u
from tests.base_tests import (
    EmptyColumnsFailTests,
    OtherBaseBehaviourTests,
)
from tests.numeric.test_BaseNumericTransformer import (
    BaseNumericTransformerFitTests,
    BaseNumericTransformerInitTests,
    BaseNumericTransformerTransformTests,
)
from tubular.numeric import OneDKmeansTransformer


class TestInit(
    BaseNumericTransformerInitTests,
    EmptyColumnsFailTests,
):
    """Tests for OneDKmeansTransformer.init()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneDKmeansTransformer"

    def test_clone(self):
        pass

    def test_acolumns_type_error_if_not_str_or_len1_list(self):
        """Test that an exception is raised if kmeans_kwargs is not a dict."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            OneDKmeansTransformer(columns=["b", "c"], new_column_name="a")

    def test_min_explained_dispersion_type_error(self):
        """Test that an exception is raised if min_explained_dispersion is not a float."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
        ):
            OneDKmeansTransformer(
                columns="b",
                new_column_name="a",
                min_explained_dispersion="0.9",
            )

    @pytest.mark.parametrize("bad_value", [-0.1, 1.1])
    def test_min_explained_dispersion_value_error(self, bad_value):
        """Test that an exception is raised if min_explained_dispersion is outside [0, 1]."""
        with pytest.raises(
            ValueError,
            match="min_explained_dispersion must be between 0 and 1",
        ):
            OneDKmeansTransformer(
                columns="b",
                new_column_name="a",
                min_explained_dispersion=bad_value,
            )

    def test_n_clusters_value_error(self):
        """Test that an exception is raised if n_clusters is below 1."""
        with pytest.raises(
            ValueError,
            match="n_clusters must be at least 1",
        ):
            OneDKmeansTransformer(
                columns="b",
                new_column_name="a",
                n_clusters=0,
            )


class TestFit(BaseNumericTransformerFitTests):
    """Tests for OneDKmeansTransformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneDKmeansTransformer"

    # This is dealt by @deartype
    def test_unexpected_kwarg_error(self):
        pass

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("df_generator"),
        [
            (d.create_df_9),  # int with None
            (d.create_bool_and_float_df),  # float with np.nan
            (d.create_df_with_none_and_nan_cols),  # all np.nan
        ],
    )
    def test_x_nans_value_type_error(
        self,
        library,
        df_generator,
    ):
        """Test that an exception is raised if X contains Nan or None."""

        df = df_generator(library=library)
        transformer = OneDKmeansTransformer(
            new_column_name="b",
            columns="a",
        )

        with pytest.raises(
            ValueError,
            match=r"OneDKmeansTransformer: X should not contain missing values.",
        ):
            transformer.fit(X=df)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_fixed_bucket_count_without_threshold(self, library):
        """Test that exactly n_clusters buckets are formed when min_explained_dispersion is not set."""
        df = u.dataframe_init_dispatch(
            {"a": [0.0, 0.1, -0.1, 100.0, 100.1, 99.9, 200.0, 200.1, 199.9]},
            library,
        )
        transformer = OneDKmeansTransformer(
            columns="a",
            new_column_name="b",
            n_clusters=2,
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        assert len(transformer.centers_) == transformer.n_clusters

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("threshold", "expected_n_buckets"),
        [
            (0.5, 2),
            (0.99, 3),
        ],
    )
    def test_smallest_sufficient_bucket_count_selected(
        self,
        library,
        threshold,
        expected_n_buckets,
    ):
        """Test that fit selects the smallest bucket count explaining enough of the dispersion."""
        df = u.dataframe_init_dispatch(
            {"a": [0.0, 0.1, -0.1, 100.0, 100.1, 99.9, 200.0, 200.1, 199.9] * 4},
            library,
        )
        transformer = OneDKmeansTransformer(
            columns="a",
            new_column_name="b",
            n_clusters=8,
            min_explained_dispersion=threshold,
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        assert len(transformer.centers_) == expected_n_buckets
        assert transformer.explained_dispersion_ >= threshold

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_bucket_count_capped_when_threshold_unreachable(self, library):
        """Test that n_clusters caps the bucket count when the threshold cannot be reached."""
        df = u.dataframe_init_dispatch(
            {"a": [0.0, 0.1, -0.1, 100.0, 100.1, 99.9, 200.0, 200.1, 199.9]},
            library,
        )
        transformer = OneDKmeansTransformer(
            columns="a",
            new_column_name="b",
            n_clusters=2,
            min_explained_dispersion=1.0,
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        assert len(transformer.centers_) == transformer.n_clusters
        assert transformer.explained_dispersion_ < 1.0

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_bucket_count_capped_at_distinct_values(self, library):
        """Test that buckets never exceed the number of distinct values, even if n_clusters does."""
        df = u.dataframe_init_dispatch(
            {"a": [1, 1, 1, 2, 2, 3]},
            library,
        )
        transformer = OneDKmeansTransformer(
            columns="a",
            new_column_name="b",
            n_clusters=8,
        ).fit(df)

        assert transformer.centers_ == [1.0, 2.0, 3.0]
        # duplicate values each count towards their bucket population
        assert transformer.bucket_population_ == [3, 2, 1]

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_fit_stores_centres_and_population(self, library):
        """Test that fit stores the selected centres and the effective population of each bucket."""
        df = u.dataframe_init_dispatch(
            {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
            library,
        )
        transformer = OneDKmeansTransformer(
            columns="a",
            n_clusters=2,
            new_column_name="new",
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        assert transformer.centers_ == [2.0, 4.0]
        assert transformer.bucket_population_ == [3, 1]
        assert transformer.explained_dispersion_ == pytest.approx(0.6)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_population_sums_to_row_count(self, library):
        """Test that the bucket populations add up to the number of fit rows."""
        df = u.dataframe_init_dispatch(
            {"a": [4, 5, 4, 5, 2, 1, 3, 2, 1, 5]},
            library,
        )
        transformer = OneDKmeansTransformer(
            columns="a",
            new_column_name="b",
            n_clusters=3,
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        assert sum(transformer.bucket_population_) == len(df)
        assert len(transformer.bucket_population_) == len(transformer.centers_)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_constant_column_gives_single_bucket(self, library):
        """Test that a constant column is summarised by one bucket explaining everything."""
        df = u.dataframe_init_dispatch(
            {"a": [7, 7, 7, 7]},
            library,
        )
        transformer = OneDKmeansTransformer(
            columns="a",
            new_column_name="b",
            n_clusters=5,
            min_explained_dispersion=0.9,
        ).fit(df)

        assert transformer.centers_ == [7.0]
        assert transformer.bucket_population_ == [4]
        assert transformer.explained_dispersion_ == pytest.approx(1.0)


# Create test data
def create_numeric_df_1(library="pandas"):
    """Example with numeric dataframe."""

    df_dict = {
        "a": [4, 5, 4, 5, 2, 1, 3, 2, 1, 5],
        "b": [43, 77, 61, 29, 84, 29, 24, 40, 84, 96],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def expected_numeric_df_1(library="pandas"):
    """Example with numeric dataframe."""

    df_dict = {
        "a": [4, 5, 4, 5, 2, 1, 3, 2, 1, 5],
        "b": [43, 77, 61, 29, 84, 29, 24, 40, 84, 96],
        "new": [0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def create_numeric_df_2(library="pandas"):
    """Example with numeric dataframe."""

    df_dict = {
        "a": [4, 5, 4, 5, 2, 1, 3, 2, 1, 5, 10, 12, 4, 16, 17],
        "b": [43, -77, -61, 29, 84, 29, -24, 40, 84, -96, 10, -4, 15, -12, 15],
        "c": [
            "a",
            "b",
            "a",
            "b",
            "a",
            "b",
            "b",
            "a",
            "c",
            "b",
            "a",
            "c",
            "a",
            "c",
            "a",
        ],
    }

    return u.dataframe_init_dispatch(df_dict, library)


def expected_numeric_df_2(library="pandas"):
    """Example with numeric dataframe."""

    df_dict = {
        "a": [4, 5, 4, 5, 2, 1, 3, 2, 1, 5, 10, 12, 4, 16, 17],
        "b": [43, -77, -61, 29, 84, 29, -24, 40, 84, -96, 10, -4, 15, -12, 15],
        "c": [
            "a",
            "b",
            "a",
            "b",
            "a",
            "b",
            "b",
            "a",
            "c",
            "b",
            "a",
            "c",
            "a",
            "c",
            "a",
        ],
        "new": [3, 0, 0, 3, 4, 3, 1, 3, 4, 0, 2, 1, 2, 1, 2],
    }

    return u.dataframe_init_dispatch(df_dict, library)


class TestTransform(
    BaseNumericTransformerTransformTests,
):
    """Tests for OneDKmeansTransformer.transform()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneDKmeansTransformer"

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                create_numeric_df_1(library="pandas"),
                expected_numeric_df_1(library="pandas"),
            ),
            (
                create_numeric_df_1(library="polars"),
                expected_numeric_df_1(library="polars"),
            ),
        ],
    )
    def test_expected_output_without(self, df, expected, from_json):
        """Test that the output is expected from transform, when there are no negative numbers"""
        x = OneDKmeansTransformer(
            columns="b",
            n_clusters=2,
            new_column_name="new",
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        x = u._handle_from_json(x, from_json)

        df_transformed = x.transform(df)

        u.assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                create_numeric_df_1(library="pandas"),
                expected_numeric_df_1(library="pandas"),
            ),
            (
                create_numeric_df_1(library="polars"),
                expected_numeric_df_1(library="polars"),
            ),
        ],
    )
    def test_expected_output(self, df, expected, from_json):
        """Test that the output is expected from transform, when there are no negative numbers"""
        x = OneDKmeansTransformer(
            columns="b",
            n_clusters=2,
            new_column_name="new",
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        x = u._handle_from_json(x, from_json)
        df_transformed = x.transform(df)

        u.assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize(
        ("df", "expected"),
        [
            (
                create_numeric_df_2(library="pandas"),
                expected_numeric_df_2(library="pandas"),
            ),
            (
                create_numeric_df_2(library="polars"),
                expected_numeric_df_2(library="polars"),
            ),
        ],
    )
    def test_expected_output_negatives(self, df, expected, from_json):
        """Test that the output is expected from transform, when there are negative numbers"""
        x = OneDKmeansTransformer(
            columns="b",
            n_clusters=5,
            new_column_name="new",
            kmeans_kwargs={"random_state": 42},
        ).fit(df)
        x = u._handle_from_json(x, from_json)
        df_transformed = x.transform(df)

        u.assert_frame_equal_dispatch(expected, df_transformed)

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_tie_assigned_to_lower_bucket(self, library):
        """Test that a value exactly halfway between two centres goes to the lower bucket."""
        df = u.dataframe_init_dispatch(
            {"a": [0, 0, 0, 10, 10, 10]},
            library,
        )
        x = OneDKmeansTransformer(
            columns="a",
            n_clusters=2,
            new_column_name="new",
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        assert x.centers_ == [0.0, 10.0]

        to_score = u.dataframe_init_dispatch({"a": [5]}, library)
        scored = x.transform(to_score)

        u.assert_frame_equal_dispatch(
            u.dataframe_init_dispatch({"a": [5], "new": [0]}, library),
            scored,
        )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_duplicate_values_same_bucket(self, library):
        """Test that duplicate values are always assigned to the same bucket."""
        df = u.dataframe_init_dispatch(
            {"a": [0, 0, 0, 10, 10, 10]},
            library,
        )
        x = OneDKmeansTransformer(
            columns="a",
            n_clusters=2,
            new_column_name="new",
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        to_score = u.dataframe_init_dispatch({"a": [3, 3, 3]}, library)
        scored = x.transform(to_score)

        u.assert_frame_equal_dispatch(
            u.dataframe_init_dispatch({"a": [3, 3, 3], "new": [0, 0, 0]}, library),
            scored,
        )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_unseen_values_assigned_to_nearest_centre(self, library):
        """Test that values not seen at fit are scored by their nearest centre."""
        df = u.dataframe_init_dispatch(
            {"a": [0, 0, 0, 10, 10, 10]},
            library,
        )
        x = OneDKmeansTransformer(
            columns="a",
            n_clusters=2,
            new_column_name="new",
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        to_score = u.dataframe_init_dispatch({"a": [-3, 4, 6, 100]}, library)
        scored = x.transform(to_score)

        u.assert_frame_equal_dispatch(
            u.dataframe_init_dispatch(
                {"a": [-3, 4, 6, 100], "new": [0, 0, 1, 1]},
                library,
            ),
            scored,
        )

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_transform_uses_only_centres(self, library):
        """Test that scoring only depends on the fitted centres, not the bucket populations."""
        df = u.dataframe_init_dispatch(
            {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
            library,
        )
        x = OneDKmeansTransformer(
            columns="a",
            n_clusters=2,
            new_column_name="new",
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        expected = x.transform(df)

        # populations are review-only information, editing them must not
        # change scoring
        x.bucket_population_ = [999, 999]
        x.explained_dispersion_ = 0.0

        u.assert_frame_equal_dispatch(expected, x.transform(df))

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_json_roundtrip_preserves_scoring(self, library):
        """Test that the fitted centres and populations survive a to/from json roundtrip."""
        df = u.dataframe_init_dispatch(
            {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
            library,
        )
        x = OneDKmeansTransformer(
            columns="a",
            n_clusters=2,
            new_column_name="new",
            min_explained_dispersion=0.5,
            kmeans_kwargs={"random_state": 42},
        ).fit(df)

        json_dict = x.to_json()

        assert json_dict["init"]["min_explained_dispersion"] == pytest.approx(0.5)
        assert json_dict["fit"]["centers_"] == x.centers_
        assert json_dict["fit"]["bucket_population_"] == x.bucket_population_
        assert json_dict["fit"]["explained_dispersion_"] == x.explained_dispersion_

        rebuilt = OneDKmeansTransformer.from_json(json_dict)

        u.assert_frame_equal_dispatch(x.transform(df), rebuilt.transform(df))


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for OneDKmeansTransformer outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "OneDKmeansTransformer"

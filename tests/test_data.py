import io
import zipfile

import pandas as pd
import pytest

from banklab.data import DatasetError, find_csv_in_zip, make_synthetic_demo, validate_data
from banklab.features import chronological_indices, prepare_features


def test_demo_is_marked_schema_and_outcome():
    df = make_synthetic_demo(500, seed=7)
    validate_data(df)
    assert "duration" in df and "y" in df
    assert set(df.y.unique()) == {"yes", "no"}


def test_feature_leakage_exclusion_and_pdays_sentinel():
    df = make_synthetic_demo(300)
    X, y = prepare_features(df)
    assert "duration" not in X.columns
    assert "y" not in X.columns
    assert "pdays" not in X.columns
    assert X.loc[df.pdays.eq(999), "days_since_previous"].isna().all()
    assert len(X) == len(y)


def test_chronological_split_is_disjoint():
    a, b, c = chronological_indices(101)
    assert (a[-1] + 1 == b[0]) and (b[-1] + 1 == c[0])
    assert len(set(a) & set(b)) == 0
    assert len(a) + len(b) + len(c) == 101


def test_nested_zip_extraction():
    content = b'age;duration;campaign;pdays;previous;poutcome;y\n40;20;1;999;0;nonexistent;no\n'
    nested_buf = io.BytesIO()
    with zipfile.ZipFile(nested_buf, "w") as z:
        z.writestr("bank-additional/bank-additional-full.csv", content)
    outer_buf = io.BytesIO()
    with zipfile.ZipFile(outer_buf, "w") as z:
        z.writestr("bank-additional.zip", nested_buf.getvalue())
    assert find_csv_in_zip(outer_buf.getvalue()) == content


def test_bad_data_rejected():
    with pytest.raises(DatasetError):
        validate_data(pd.DataFrame({"x": [1]}))

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from salary_estimates.EDA import return_missing


@pytest.mark.parametrize(
    "feature_dict, col_names, missing_values_test, parent_missing_categories_test",
    [
        (
                {
                    'train': {'cat1': {'test1', 'test2'}, 'cat2': {'test3', 'test4'}},
                    'test': {
                        'cat1': {'test1', 'test2'},
                        'cat2': {'test3', 'test5'},
                        'cat3': {},
                    },
                },
                ['cat1', 'cat2', 'cat3'],
                {'test5', 'test4'},
                'cat3'
         )
    ],
)
def test_returns_differences(feature_dict, col_names, missing_values_test, parent_missing_categories_test):
    parent_missing_categories, missing_features = return_missing(feature_dict, col_names)
    assert parent_missing_categories_test == parent_missing_categories
    assert missing_values_test == missing_features

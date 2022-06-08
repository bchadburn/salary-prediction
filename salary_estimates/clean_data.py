import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from salary_estimates.utils.constants import TRAIN_FEATURES_PATH, TRAIN_SALARIES_PATH, CLEANED_DATA_PATH, TEST_FEATURES_PATH, TEST_DATA_PATH
import argparse
from sklearn.preprocessing import OrdinalEncoder


def add_dummy_columns(data, feature_col):
    dummy_tbl = pd.get_dummies(data[[feature_col]])
    final_tbl = pd.concat([data, dummy_tbl], axis=1)
    return final_tbl


def add_ordinal_encoding(data, col, value_order):
    ordinal_encoder = OrdinalEncoder(categories=[value_order])
    values = data[col].values.reshape(-1, 1)
    transformed_values = ordinal_encoder.fit_transform(values)
    data[f'{col}_encoded'] = transformed_values
    return data


def main(data_destination, test_data_destination):
    train_features = pd.read_csv(TRAIN_FEATURES_PATH)
    train_salaries = pd.read_csv(TRAIN_SALARIES_PATH)
    test_features = pd.read_csv(TEST_FEATURES_PATH)

    train_features.set_index('jobId', inplace=True)
    train_salaries.set_index('jobId', inplace=True)
    test_features.set_index('jobId', inplace=True)

    original_categorical_columns = train_features.columns.tolist()[:-2]
    categorical_feature_columns = original_categorical_columns[1:]  # Drop companyId for now

    # Create dummy columns
    cleaned_log_data = train_features.copy()
    test_cleaned_log_data = test_features.copy()
    degree_order = ['NONE', 'HIGH_SCHOOL', 'BACHELORS', 'MASTERS', 'DOCTORAL']

    for category in categorical_feature_columns:
        if category != 'degree':
            cleaned_log_data = add_dummy_columns(cleaned_log_data, category)
            test_cleaned_log_data = add_dummy_columns(test_cleaned_log_data, category)
        else:
            cleaned_log_data = add_ordinal_encoding(cleaned_log_data, category, degree_order)
            test_cleaned_log_data = add_ordinal_encoding(test_cleaned_log_data, category, degree_order)

    # Merge salaries and drop unneeded columns and incorrect values
    train_merge = cleaned_log_data.merge(train_salaries, left_index=True, right_index=True)
    train_merge = train_merge[train_merge['salary'] > 0]  # Removes salaries set to 0

    # Compute salaries using all data for the test data.
    # For the training data we have to add company avg sale price after the train/test split
    company_avg_salaries = train_merge.groupby(['companyId', 'jobType'])['salary'].mean()
    company_avg_salaries.rename('companyId', inplace=True)

    # Add avg company salary feature to test data
    test_cleaned_log_data = test_cleaned_log_data.merge(company_avg_salaries, left_on=['companyId', 'jobType'],
                                                        right_index=True)
    test_cleaned_log_data['company_avg_salary'] = test_cleaned_log_data['companyId_y']
    test_cleaned_log_data.drop(columns=['companyId_x', 'companyId_y'], inplace=True)

    # Drop unneeded columns
    train_merge.drop(columns=categorical_feature_columns[1:], inplace=True)  # Keep companyId, remove before model fitting.
    test_cleaned_log_data.drop(columns=original_categorical_columns, inplace=True)

    train_merge.to_csv(data_destination)
    test_cleaned_log_data.to_csv(test_data_destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleaned_data_destination', type=str, default=CLEANED_DATA_PATH)
    parser.add_argument('--test_data_destination', type=str, default=TEST_DATA_PATH)
    args, _ = parser.parse_known_args()

    main(args.cleaned_data_destination, args.test_data_destination)
    print('Cleaned and saved data set')

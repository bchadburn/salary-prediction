import pandas as pd
from salary_estimates.utils.constants import TRAIN_FEATURES_PATH, TRAIN_SALARIES_PATH, TEST_FEATURES_PATH
import matplotlib.pyplot as plt


def return_avg_salary(data, group, salary_col):
    print('mean salary:', data.groupby(group, as_index=False)[salary_col].mean())


def create_salary_reports(data, salary_col, limit=20):
    for group in data.columns.tolist():
        if len(set(data[group])) < limit:
            data.boxplot(column=[salary_col], by=[group])
            plt.show()
            return_avg_salary(data, group, salary_col)
            print('mean salary:', data.groupby(group, as_index=False)[salary_col].mean())


def assign_features(data: pd.DataFrame, data_type: str, column_names: list, feature_dict: dict) -> dict:
    for category in column_names:
        feature_dict[data_type][category] = set(data[category])
    return feature_dict


def return_missing(feature_dict: dict, column_names: list):
    missing_parent_categories = {}
    missing_features = []
    for category in column_names:
        if category not in feature_dict['train']:
            missing_parent_categories['train'] = category
        elif category not in feature_dict['test']:
            missing_parent_categories['test'] = category
        else:
            missing_features = feature_dict['train'][category].symmetric_difference(feature_dict['test'][category])
    if len(missing_features) > 0:
        print(f'found {len(missing_features)} unique categories. Items: {missing_features}')
    if len(missing_parent_categories) > 0:
        print(f'found {len(missing_parent_categories)} unique parent categories. Items: {missing_parent_categories}')
        return category, missing_features


def main():
    # Read in data
    train_features = pd.read_csv(TRAIN_FEATURES_PATH)
    train_salaries = pd.read_csv(TRAIN_SALARIES_PATH)

    train_features[['yearsExperience', 'milesFromMetropolis']].describe()

    # Review distribution of numeric features
    train_features['yearsExperience'].plot(kind='hist')
    plt.show()

    train_features['milesFromMetropolis'].plot(kind='hist')
    plt.show()

    # Review counts
    categories = train_features.columns.tolist()[1:-2]
    for category in categories:
        print(train_features.groupby(category).size())
        print("\n")
    # We see large number of samples for each category for each column.

    # Add salary to create correlation and box plots
    train_merge = train_features.merge(train_salaries, on='jobId')

    # Compute Correlations
    year_corr = train_merge['yearsExperience'].corr(train_merge['salary'])
    print('yearsExperience correlation with salary:', year_corr)

    year_corr = train_merge['milesFromMetropolis'].corr(train_merge['salary'])
    print('milesFromMetropolis correlation with salary:', year_corr)
    # We see yearsExperience is positively correlated with salary while milesFromMetropolis is negatively correlated

    # Print means and create boxplots for categorical features
    create_salary_reports(train_merge, 'salary', 20)

    # We can see fairly large differences between categories and a few trends
    # jobtype: CEO has highest avg pay followed by CFO, CTO, lowers Janitor and Junior. Senior and manger have fairly similar pay and distributions
    # Education: None shows lowest pay then high school although there is a lot of variability. Highest is Doctoral
    # Major: Engineering is highest, 'None' is lowest. Biology and Chemistry is similar and may be able to grouped together.
    # Industry: Oil and Finance is highest and education is lowest.

    # Check what categories are different from train and test set e.g. if COO is in training but not in test
    test_features = pd.read_csv(TEST_FEATURES_PATH)
    unique_categories = {'train': {}, 'test': {}}

    unique_categories = assign_features(train_features, 'train', categories, unique_categories)
    unique_categories = assign_features(test_features, 'test', categories, unique_categories)

    return_missing(unique_categories, categories)
    # We aren't seeing any new categories between our training and test set


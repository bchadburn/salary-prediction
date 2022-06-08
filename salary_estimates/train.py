import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from salary_estimates.utils.constants import CLEANED_DATA_PATH, DATA_DIRECTORY
from salary_estimates.utils.utils import time_run, verify_create_paths


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def add_avg_company_salary(train, test, train_salary):
    """Performs target encoding. Calculates avg company salary from training set and
    provides it as a feature for both train and test data"""
    # Add original sort index
    train['sort'] = range(1, len(train) + 1)
    test['sort'] = range(1, len(test) + 1)

    train_salary_data = train.merge(train_salary, left_index=True, right_index=True)
    train_company_avg_salaries = train_salary_data.groupby(['companyId', 'jobType'])['salary'].mean()
    train = train.merge(train_company_avg_salaries.rename('companyId'), left_on=['companyId', 'jobType'], right_index=True)
    test = test.merge(train_company_avg_salaries.rename('companyId'), left_on=['companyId', 'jobType'], right_index=True)

    train['company_avg_salary'] = train['companyId_y']
    test['company_avg_salary'] = test['companyId_y']

    # Reset to original sort
    train.sort_values(by='sort', ascending=True, inplace=True)
    test.sort_values(by='sort', ascending=True, inplace=True)

    train.drop(columns=['companyId_x', 'companyId_y', 'companyId', 'sort', 'jobType'], inplace=True)
    test.drop(columns=['companyId_x', 'companyId_y', 'companyId', 'sort', 'jobType'], inplace=True)
    return train, test


@time_run
def main(val_size, subset_data, eta, max_depth, n_estimators, min_child_weight, model_folder, experiment_id):
    train_features = pd.read_csv(CLEANED_DATA_PATH, index_col='jobId')
    # train_features.drop(columns=['companyId', 'jobType'], inplace=True)
    train_features = train_features[:int(round(len(train_features)*subset_data, 0))]
    X_train, X_test, y_train, y_test = train_test_split(train_features.iloc[:, :-1],
                                                        train_features.salary,
                                                        test_size=val_size, random_state=99)
    X_train, X_test = add_avg_company_salary(X_train, X_test, y_train)

    # # Grid Search XGBoost
    # search_params = {'n_estimators': [1300], 'max_depth': [2], 'eta': [.4], 'alpha': [0]}
    #
    # grid_search = GridSearchCV(estimator=XGBRegressor(max_depth=2, eta=0.3, alpha=0, n_estimators=1000),
    #                            n_jobs=2,
    #                            param_grid=search_params,
    #                            pre_dispatch='2*n_jobs',
    #                            cv=5)
    #
    # grid_search.fit(X_train, y_train)
    # grid_search.best_params_
    #
    # ## Fit final model
    # model = XGBRegressor(max_depth=2, eta=0.4, alpha=0, n_estimators=1300)
    # model.fit(X_train, y_train)
    #
    # # Predict and evaluate model
    # y_pred = model.predict(X_test)
    # rmse, mae, r2 = eval_metrics(y_test, y_pred)
    # print('rmse:', rmse, 'mae:', mae, 'r2:', r2)
    #
    # # Plot Feature important graph
    # sorted_idx = model.feature_importances_.argsort()
    # plt.gcf().subplots_adjust(bottom=0.40)
    # sns.barplot(X_train.columns[sorted_idx], model.feature_importances_[sorted_idx])
    # plt.title("Feature Importance")
    # plt.xticks(rotation=90)
    # plt.savefig('feature_importance.png')
    # plt.show()

    # Use mlflow to track experiments
    run_name = f"XGBoost {experiment_id} | {val_size * 100}% train set"
    verify_create_paths(os.path.join('mlruns', str(experiment_id)))
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.set_tag("Features", train_features.columns[:-1])

        reg = XGBRegressor(max_depth=max_depth, eta=eta, min_child_weight=min_child_weight, n_estimators=n_estimators)
        model = reg.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        rmse, mae, r2 = eval_metrics(y_train, y_pred)
        print('training performance')
        print('rmse:', rmse, 'mae:', mae, 'r2:', r2)

        y_pred = model.predict(X_test)

        rmse, mae, r2 = eval_metrics(y_test, y_pred)
        print('test performance')
        print('rmse:', rmse, 'mae:', mae, 'r2:', r2)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("eta", eta)
        mlflow.log_metric(key="rmse", value=rmse)
        mlflow.log_metric(key="train_mae", value=mae)
        mlflow.log_metric(key="train_r2_score", value=r2)
        model_path = os.path.join(f"{model_folder}", f"model-{run_id}-{n_estimators}-{max_depth}-{eta}")
        mlflow.sklearn.save_model(model, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_size', type=float, default=.2)
    parser.add_argument('--subset_data', type=float, default=1)
    parser.add_argument('--eta', type=float, default=.4)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--n_estimators', type=int, default=1200)
    parser.add_argument('--min_child_weight', type=int, default=1)
    parser.add_argument('--model_folder', type=str, default=DATA_DIRECTORY)
    parser.add_argument('--experiment_id', type=int, default=0)
    args, _ = parser.parse_known_args()

    main(args.val_size, args.subset_data, args.eta, args.max_depth,
         args.n_estimators, args.min_child_weight, args.model_folder, args.experiment_id)
    print('Saved model predictions on test set')

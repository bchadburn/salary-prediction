import pandas as pd
import mlflow
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from salary_estimates.utils.constants import TEST_DATA_PATH, PREDICTIONS_PATH, MODEL_FOLDER


def main(test_data_path, pred_path, model_path):
    test_features = pd.read_csv(test_data_path, index_col='jobId')
    loaded_model = mlflow.sklearn.load_model(model_path)

    # Predict and evaluate model
    y_pred = loaded_model.predict(test_features)
    predictions = pd.DataFrame(list(zip(test_features.index.values.tolist(), pd.Series(y_pred))),
                               columns=['jobId', 'salary'])
    predictions.to_csv(pred_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, default=TEST_DATA_PATH)
    parser.add_argument('--predictions_destination', type=str, default=PREDICTIONS_PATH)
    parser.add_argument('--model_folder', type=str, default=MODEL_FOLDER)
    args, _ = parser.parse_known_args()

    main(args.test_data_path, args.predictions_destination, args.model_path)
    print('Saved model predictions on test set')

# includes for main
import argparse
import os
import pandas as pd

from sklearn.metrics import classification_report

# from this project
from data_prepare import DataPrepare
from feature_engineering import FeatureEngineering
import utils.helper_models as helper_models


class TestPipeline:

    def __init__(self, input_csv_directory_path, input_csv_file_name):

        self.input_csv_directory_path=input_csv_directory_path
        self.input_csv_file_name = input_csv_file_name

    def fit_pipeline(self, input_csv_directory_path, input_csv_file_name):

        print('Start Testing pipeline')

        target_name = 'income'

        try:

            data_test = pd.read_csv(os.path.join(input_csv_directory_path, input_csv_file_name))

            print(data_test.head())
            print(data_test.shape)

            data_prepare = DataPrepare()
            df_clean = data_prepare.dataPrepare(data_test)

            feature_engineering = FeatureEngineering()
            df_features_target = feature_engineering.featureEngineering(df_clean)

            # Dropping missing values if any
            df_features_target.dropna(axis=0, inplace=True)

            model_pipeline = helper_models.load_pipeline()

            prediction = model_pipeline.predict(df_features_target.drop(columns=target_name))
            probability = model_pipeline.predict_proba(df_features_target.drop(columns=target_name))

            print("Classification report: \n ", classification_report(df_features_target[target_name],
                                                                 prediction))

            helper_models.fill_confusion_matrix_and_save(df_features_target[target_name],prediction,
                                                         f_name='Test Confusion matrix',
                                                         out_dir=input_csv_directory_path)

            helper_models.plot_roc_curve_and_save(df_features_target[target_name],
                                                  probability, f_name='Test Roc Curve',
                                                  out_dir=input_csv_directory_path)

            print('Pipeline completed successfully and results are stored in data directory')

        except Exception as ex:
            print('Something went wrong with the Pipeline %s', ex)
            raise ex




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read the raw data')
    parser.add_argument('--input-csv-directory-path', type=str, default="./",
                        help='Full path to hidden test data ')

    parser.add_argument('--input-csv-file-name', type=str, default='test_data.csv',
                        help='Name of the test file')

    opt = parser.parse_args()
    print(opt)

    test_pipeline = TestPipeline(opt.input_csv_directory_path,opt.input_csv_file_name)

    test_pipeline.fit_pipeline(opt.input_csv_directory_path, opt.input_csv_file_name)
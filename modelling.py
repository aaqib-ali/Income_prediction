import time
import os
import pandas as pd
from collections import Counter
import seaborn as sns
import utils.helper_exploration as exploration
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

# sklearn classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier


# from this project
import utils.helper_models as helper_models


class Modelling:

    def modelling(self, df_features_target):
        print('Start Modelling:')

        #model_name = 'LogisticRegression'
        #model_name = 'SGDClassifier'
        #model_name = 'RandomForest'
        model_name = 'XgBoost'


        # Checking missing values before model implementation
        data_check = helper_models.missing_values_table(df_features_target)
        print('Missing values in a column with the percentage', data_check)

        # Split the data into train and Test
        train_data_df, test_data_df = helper_models.split_dataset(df_features_target)

        categorical_features, numeric_features, target_name = helper_models.read_features_name()

        #train_data_df = helper_models.donwsample_trainset(train_data_df, target_name, 1234)

        params = helper_models.read_parameters(model_name)
        print(*params)

        #classifier = LogisticRegression(**params)
        #classifier = SGDClassifier(**params)
        #classifier = RandomForestClassifier(**params)
        classifier = XGBClassifier(**params)

        print('XGBoost is Initialized')

        #Build Model pipeline
        categorical_pipe = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        numerical_pipe = Pipeline([
            ('Scaler', StandardScaler())
        ])
        preprocessing = ColumnTransformer(
            [('cat', categorical_pipe, categorical_features),
                ('num', numerical_pipe, numeric_features)])

        model_pipeline = Pipeline([
            ('preprocess', preprocessing),
            ('classifier', classifier)])

        print('Total Targets shape %s' % Counter(df_features_target[target_name]))
        print('Train Targets shape %s' % Counter(train_data_df[target_name]))
        print('Test Targets shape %s' % Counter(test_data_df[target_name]))

        # fit the model
        model_pipeline.fit(train_data_df.drop(columns=target_name), train_data_df[target_name])

        helper_models.save_pipeline(model_pipeline)

        #'''
        prediction = model_pipeline.predict(test_data_df.drop(columns=target_name))
        probability = model_pipeline.predict_proba(test_data_df.drop(columns=target_name))

        print("Classification report", classification_report(test_data_df[target_name], prediction))

        helper_models.fill_confusion_matrix_and_save(test_data_df[target_name],
                                                         prediction,f_name='Confusion matrix '+
                                                                           str(model_name), out_dir='./results/Model_performance')

        helper_models.plot_roc_curve_and_save(test_data_df[target_name],
                                                  probability, f_name='Roc Curve '+
                                                                      str(model_name),out_dir='./results/Model_performance')

        helper_models.plot_feature_importance_and_save(model_pipeline, categorical_features,
                                                       numeric_features, top_num=10,
                                                       f_name='Random Forest Classifier Feature Importance',
                                                       out_dir='./results/Model_performance')
        #'''



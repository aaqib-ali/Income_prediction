import os
import pandas as pd

# from this project
import utils.common as common


class FeatureEngineering:

    def featureEngineering(self, df_clean):

        print('Start Feature Engineering:')

        df_clean = common.fill_missing_values(df_clean)

        # removing columns bases upon pandas profiling report

        # Remove fnlwgt : Because No correction with income, It doesn't represent particular individual.
        # Remove education: Because education_num represents this ordinal column well.
        # Remove workclass: Because of high correlation and also occupation has more strong correlation with income.
        # Remove native_country: Because of not a strong correlation we want to reduce the chance of overfitting.
        # Remove marital_status: Because of not a strong correlation we want to reduce the chance of overfitting.

        cols_to_drop = ['fnlwgt', 'education', 'workclass', 'native_country', 'marital_status', 'sex']
        df_features_target = common.remove_cols(df_clean, cols_to_drop)
        #print(df_clean.columns)

        print('Extract features Succesfully...')

        return df_features_target
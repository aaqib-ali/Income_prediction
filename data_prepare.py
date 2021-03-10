import time
import numpy as np
import os
import pandas as pd

# from this project


class DataPrepare:

    def dataPrepare(self, df_raw_data):
        print('Start Data prepration:')

        start = time.time()

        # cleaning this up and removing the period from those stray values so our values line up better
        df_raw_data['income'] = df_raw_data['income'].str.replace('.', '', regex=False)
        df_raw_data['income'] = df_raw_data['income'].str.replace(' ', '', regex=False)

        # Convert our actual Target
        df_raw_data['income'] = df_raw_data['income'].map(lambda x: 1 if x == '>50K' else 0)

        df_raw_data = df_raw_data.replace(' ?', np.nan)  # Convert all the outliers to missing values we will fill
                                                            # these later

        end = time.time()
        print(end - start)

        return df_raw_data

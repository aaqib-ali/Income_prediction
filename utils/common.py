import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import pandas_profiling as pdp

# Method to remove coloumns by list of names
def remove_cols(df, cols_to_drop):

    df = df.drop(columns=cols_to_drop, inplace=False)

    return df

# Method to save the plot to the directory
def save_fig(f_name, dir_name):
    plt.figure(f_name)
    plt.savefig(os.path.join(dir_name, "_".join( f_name.split() ) + ".png"), transparent=True)
    plt.close()

def my_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print("Created output dir", dir_name)

def profiling(df):

    print('Start Profiling')
    start = time.time()
    sample_for_profiling = df
    profile_target = pdp.ProfileReport(sample_for_profiling)
    print("Profile", time.time() - start, "s")
    profile_target.to_file("profile_target.html")

def fill_missing_values(df):

    print('Missing sum before filling ', df.isna().sum())

    columns_with_missing_values = ['workclass', 'occupation', 'native_country']

    # We are filling missing values with the mode of the column we can also train a simple model
    # to predict and fill these missing values (Time Constraint)

    for col in columns_with_missing_values:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df



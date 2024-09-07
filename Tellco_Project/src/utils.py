import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import zscore

def missing_value_table(df):
    # Total missing value
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # datatype of missing values
    mis_val_dtype = df.dtypes
    
    # Make table with the result
    mis_val_table = pd.concat([mis_val, mis_val_percent,mis_val_dtype], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1:'% of Total values', 2:'Dtype'})
    
    # Sort the table by the percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] !=0].sort_values(
            '% of Total Values', ascending=False).round(1)
    # Print Summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) + 
          " columns that have missing values.")
    
    # Return the dataframe with missing info
    return mis_val_table_ren_columns


def convert_bytes_to_megabytes(df, bytes_data):
    megabyte = 1 * 10e+5
    df[bytes_data] = df[bytes_data] / megabyte
    return df[bytes_data]


def fix_outlier(df, column):
    df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].median(), df[column])
    return df[column]


def remove_outliers(df, column_to_process, z_threshold=3):
    # Apply outlier removal to the specified column
    z_scores = zscore(df[column_to_process])
    outlier_column = column_to_process + '_Outlier'
    df[outlier_column] = (np.abs(z_scores) > z_threshold).astype(int)
    df = df[df[outlier_column] == 0] # keep the row without outliers 
    
    
    # Drop the outlier column as it's no longer needed
    df = df.drop(columns=[outlier_column], errors='ignore')
    
    return df
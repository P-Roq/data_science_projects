import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split as split


class Filtered_DF:
    def __init__(self):
        self.filtered_dict: dict = {} # {query_: filtered_df}

    def insert_filtered_df(self, query_: str, df: pd.core.frame.DataFrame):
        filtered = df.query(query_)
        self.filtered_dict[query_] = filtered

        return 


# Convert into binary.
def convert_binary(list_vars: list, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    
    for var in list_vars:
        if var == 'sex':
            df['sex'] = df['sex'].apply(lambda x: 0 if x=='female' else 1)
        if var == 'smoker':
           df['smoker'] = df['smoker'].apply(lambda x: 0 if x=='no' else 1)

    return df


# Country regions turned into categories.
def categorize_var(var: str, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df[var] = pd.Categorical(df[var])    
    
    return df 


# Log variables.
def log_vars(list_vars: list, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    for var in list_vars:
        logvar_name = f'log_{var}'
        df[logvar_name] = np.log(df[var])

    return df

# Split data.
def split_data(df: pd.core.frame.DataFrame, rand_state: int, test_size_: float):
    train, test = split(df, random_state=rand_state, test_size=test_size_)
    return train, test
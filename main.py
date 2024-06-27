import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import set_config
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import pickle
from tkinter import *
warnings.filterwarnings('ignore')


def read_csv(file_path):
    
    # Read data from a CSV file and return a pandas DataFrame.
    
    return pd.read_csv(file_path)


def dataset_info_statistics(data):
    # Display information and basic statistics about the dataset.

    # Display general information about the dataset
    print("Dataset Information:")
    print(data.info())
    print("\n")

    # Display basic statistics for numerical columns
    print("Basic Statistics for Numerical Columns:")
    print(data.describe())
    print("\n")

def check_null(data):

    # Check for null values in the dataset.

    null_counts = data.isnull().sum()
    print("Null Values in the Dataset:")
    return null_counts


#4.check for duplicated rows in the dataset
def check_duplicates(data):

    # Check for duplicated rows in the dataset.
    return data.duplicated().any()


#5. getting basic analysis for numerical and categorical columns
def plot_graph(data):
    # Plot graphs for numerical and categorical data in a dataframe.
    
    numerical_columns = data.select_dtypes(include=np.number).columns
     
    for column in numerical_columns:
        plt.figure(figsize=(5,3))
        sns.displot(data[column],kde=True)
        plt.title(f"Histogram for {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()
        
    categorical_columns = data.select_dtypes(include='object').columns
    for column in categorical_columns:
        plt.figure(figsize=(5, 3))
        sns.countplot(data[column])
        plt.title(f'Countplot for {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

#6. Seperate feature and target
def seperate_features_target(data,target_column):

    # Separate features and target variable
    
    X = data.drop(columns=[target_column],axis=1)
    y = data[target_column]
    
    return X,y

#7. Train test split
def perform_train_test_split(X, y, test_size=0.20, random_state=42):
    # Perform train-test split on the dataset.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test





calories = read_csv('./dataset/calories.csv')
exercise = read_csv('./dataset/exercise.csv')


data = pd.merge(calories, exercise, on='User_ID')

# print(data.head())

# print(dataset_info_statistics(data))

# plot_graph(data)

# print(check_null(data))

# print(data.columns)


X,y = seperate_features_target(data,'Calories')
X = X.drop(columns=['User_ID'])
X_train,X_test,y_train,y_test = perform_train_test_split(X, y, test_size=0.20, random_state=42)




# Column Transformer and Pipeline


preprocessor = ColumnTransformer(transformers=[
    ('ordinal',OrdinalEncoder(),['Gender']),
    ('num',StandardScaler(),['Age',
                            'Height',
                            'Weight',
                            'Duration',
                            'Heart_Rate',
                            'Body_Temp']),
],remainder='passthrough')

pipeline = Pipeline([("preprocessor",preprocessor),
                     ("model",LinearRegression())
                    ])

set_config(display='diagram')

# print(pipeline)



pipeline.fit(X_train,y_train)

y_pred = pipeline.predict(X_test)

# print(r2_score(y_test,y_pred))

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')

# print(cv_results.mean())

# print(mean_absolute_error(y_test,y_pred))





def model_scorer(model_name,model):
    
    output=[]
   
    
    output.append(model_name)
    
    pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',model)])
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
    
    pipeline.fit(X_train,y_train)
    
    y_pred = pipeline.predict(X_test)
    
    output.append(r2_score(y_test,y_pred))
    output.append(mean_absolute_error(y_test,y_pred))
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')
    output.append(cv_results.mean())
    
    return output


model_dict={
    'log':LinearRegression(),
    'RF':RandomForestRegressor(),
    'XGBR':XGBRegressor(),
}


model_output=[]
for model_name,model in model_dict.items():
    model_output.append(model_scorer(model_name,model))

# print(model_output)


preprocessor = ColumnTransformer(transformers=[
    ('ordinal',OrdinalEncoder(),['Gender']),
    ('num',StandardScaler(),['Age',
                            'Height',
                            'Weight',
                            'Duration',
                            'Heart_Rate',
                            'Body_Temp']),
    
],remainder='passthrough')


pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',XGBRegressor())
    
])


pipeline.fit(X,y)

# print(pipeline.fit(X,y))

# sample = pd.DataFrame({
#    'Gender':'female',
#     'Age':31,
#     'Height':148,
#     'Weight':50,
#     'Duration':8,
#     'Heart_Rate':84,
#     'Body_Temp':39.5,
# },index=[0])

# sample = pd.DataFrame({
#    'Gender':'male',
#     'Age':68,
#     'Height':190.0,
#     'Weight':94.0,
#     'Duration':29.0,
#     'Heart_Rate':105.0,
#     'Body_Temp':40.8,
# },index=[0])

# print(pipeline.predict(sample))




# SAVE the model


with open('pipeline.pkl','wb') as f:
    pickle.dump(pipeline,f)


with open('pipeline.pkl','rb') as f:
    pipeline_saved = pickle.load(f)


# result = 0
# def generate_answer():
#     result = pipeline_saved.predict(sample)[0]
#     return result


# ans = generate_answer()
# print(ans)
# print(result)



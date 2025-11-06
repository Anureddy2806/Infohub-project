# Create your views here.
from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import Flight_Ticket_Prediction
from django.conf import settings
import pandas as pd
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Create your views here.


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not Yet Activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):

    return render(request, 'users/UserHomePage.html', {})
#===========================================================================

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'Data_Train.csv'
    df = pd.read_csv(path, nrows=10000)
    df_html = df.to_html()  # Invoke the to_html method
    return render(request, 'users/viewdataset.html', {'data': df_html})



#===============================================================================
# EDA Libraries
# import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.subplots as sp
import plotly.graph_objects as go
from xgboost import XGBRegressor
# Data Preprocessing Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, LabelEncoder
from category_encoders import BinaryEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
# Machine Learning (regression models) Libraries
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from django.shortcuts import render
import os


def ml(request):
   # Reading csv file
    df_data = settings.MEDIA_ROOT + "//" + 'Data_Train.csv'
    df = pd.read_csv(df_data)
    df.head(10)

    # Dropping any duplicates
    df = df.drop_duplicates()

    def process_additional_info(info):
        if info == 'No info' or info == 'No Info':
            return 'No'
        else:
            return 'Yes'

    df['Has_Additional_Info'] = df['Additional_Info'].apply(process_additional_info)

    df = df.drop('Additional_Info', axis = 1)
    # Split the 'Route' column using ' → ' as the separator and extract the first part into 'Route_1'
    df['Route_1'] = df['Route'].str.split(' ? ').str[0]
    df['Route_2'] = df['Route'].str.split(' ? ').str[1]
    df['Route_3'] = df['Route'].str.split(' ? ').str[2]
    df['Route_4'] = df['Route'].str.split(' ? ').str[3]
    df[['Route_1', 'Route_2', 'Route_3', 'Route_4']] = df[['Route_1', 'Route_2', 'Route_3', 'Route_4']].fillna('None') 
    df = df.drop('Route', axis = 1)

    # Transforming 'Date_of_Journey' column from text to date format to extract day, month and year.
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d-%m-%Y')  
    # "Extracting day, month, and year from 'Date_of_Journey' for detailed time analysis." time-based analysis.
    df = df.assign( journey_day=df.Date_of_Journey.dt.day,
                journey_month=df.Date_of_Journey.dt.month,
                journey_year=df.Date_of_Journey.dt.year
                )
    df = df.drop('Date_of_Journey', axis=1)

    # Transforming 'Dep_Time' column from text to date format to extract hour and minutes
    df['Dep_Time'] = pd.to_datetime(df['Dep_Time'], format='%H:%M')
    # "Extracting hour and min from 'Dep_Time' for detailed time analysis." time-based analysis.
    df = df.assign( dep_time_hour=df.Dep_Time.dt.hour,
                dep_time_min=df.Dep_Time.dt.minute,
                )
    df = df.drop('Dep_Time', axis=1)

    df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'])
    # "Extracting hour and min from 'Arrival_Time' for detailed time analysis." time-based analysis.
    df = df.assign( Arrival_Time_hour=df.Arrival_Time.dt.hour,
                Arrival_Time_minute=df.Arrival_Time.dt.minute,
                )
    df = df.drop('Arrival_Time', axis=1)

    # Function to extract hours and minutes from a duration string
    def extract_duration(duration):
        parts = duration.split()
        hours = 0
        minutes = 0

        # Iterate through each part of the duration
        for part in parts:
            if 'h' in part:
                hours = int(part.replace('h', ''))
            elif 'm' in part:
                minutes = int(part.replace('m', ''))

        return hours, minutes

    # Creating new columns 'Duration_Hours' and 'Duration_Minutes'
    df[['Duration_hours', 'Duration_minutes']] = df['Duration'].apply(extract_duration).apply(pd.Series)

    # Dropping the 'Duration' column as we have already extracted the date features from it, and it is no longer needed for our analysis.
    df = df.drop('Duration', axis=1)

    df.dropna(axis=0, inplace=True)

    numerical_features = [
        'Price',
        'journey_day',
        'journey_month',
        'journey_year',
        'dep_time_hour',
        'dep_time_min',
        'Arrival_Time_hour',
        'Arrival_Time_minute',
        'Duration_hours',
        'Duration_minutes'
    ]

    # Detect outliers in numerical features
    # outliers_indices = detect_outliers(df, features=numerical_features, threshold=3)
    # outliers_indices = detect_outliers_zscore(df, features=numerical_features, threshold=3)
    # number_of_outliers = len(outliers_indices)

    # Print the number of outliers
    # print(f'Number of outliers: {number_of_outliers}')

    # extract the x Featues and y Label
    X = df.drop(['Price'], axis=1)
    y = df['Price']

    # Then we Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=42,
                                                        )
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

   # Define the categorical columns for binary encoding and one-hot encoding
    categorical_features_be = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Route_1', 'Route_2', 'Route_3', 'Route_4']
    categorical_features_oh = ['Has_Additional_Info']

    # Define numerical features
    numerical_features = ['journey_day', 'journey_month', 'journey_year', 'dep_time_hour', 'dep_time_min', 'Arrival_Time_hour',
    'Arrival_Time_minute', 'Duration_hours', 'Duration_minutes'
    ]


    # Define transformers for numerical and categorical features
    num_pipeline = Pipeline(steps=[
        ('scaler', RobustScaler())  # Scale numerical features using RobustScaler
    ])

    cat_pipeline_be = Pipeline(steps=[
        ('encoder', BinaryEncoder())  # Binary encode categorical features
    ])

    cat_pipeline_oh = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first'))  # One-hot encode categorical features
    ])

    # Combine numerical and categorical pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat_be', cat_pipeline_be, categorical_features_be),
            ('cat_oh', cat_pipeline_oh, categorical_features_oh),
            ('num', num_pipeline, numerical_features)
        ],
        remainder='passthrough'
    )

    regressors = [
        ("Linear Regression", LinearRegression(n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
        ("XGBoost", xgb.XGBRegressor(random_state=42, n_jobs=-1))
    ]

   # Creating lists for classifier names, mean_test_r2_scores, and results.
    results = []
    mean_test_r2_scores = []
    cross_val_errors = []
    regressor_names = []

    for model_name, model in regressors:

        # Steps Creation
        steps = [('preprocessor', preprocessor), (model_name, model)]

        # Create the pipeline
        pipeline = Pipeline(steps=steps)

        # Perform cross-validation with train scores and R2 scoring
        cv_results = cross_validate(pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1, return_train_score=True)

        # Calculate cross-validation error
        cross_val_error = 1 - np.mean(cv_results['test_score'])

        # Append results to the list
        results.append({
            "Model Name": model_name,
            "Mean Train R2 Score": np.mean(cv_results['train_score']),
            "Mean Test R2 Score": np.mean(cv_results['test_score']),
            "Cross-Validation Error": cross_val_error
        })

        mean_test_r2_scores.append(np.mean(cv_results['test_score']))
        cross_val_errors.append(cross_val_error)
        regressor_names.append(model_name)

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)

    param_grid = {
        'XGBoost__learning_rate': [0.1, 0.2, 0.3],
        'XGBoost__n_estimators': [200, 300, 400],
        'XGBoost__max_depth': [5, 6, 7]
    }

    steps=[]
    steps.append(('preprocessor', preprocessor))
    steps.append(("XGBoost", xgb.XGBRegressor(random_state=42, n_jobs=-1)))
    pipeline=Pipeline(steps=steps)

    # Create GridSearchCV instance
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1,
                            return_train_score=True)

    # Fit the pipeline with GridSearch to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    final_model=grid_search.best_estimator_

    # Evaluate the final model on the separate test set
    test_score = final_model.score(X_test, y_test)
    print("Test R2 Score on Separate Test Set:", test_score)

    # Assuming the 'Additional_Info' value for this input is 'No info'
    additional_info = 'No info'  # Replace this with the actual value from your input data

    # Create the input_data DataFrame with the required columns
    input_data = pd.DataFrame({
        'Airline': ['IndiGo'],
        'Source': ['Bangalore'],
        'Destination': ['New Delhi'],
        'Total_Stops': ['non-stop'],
        'Route_1': ['No BLR ? DEL'],
        'Route_2': ['None'],
        'Route_3': ['None'],
        'Route_4': ['None'],
        'journey_day': [240],
        'journey_month': [30],
        'journey_year': [2019],
        'dep_time_hour': [220],
        'dep_time_min': [200],
        'Arrival_Time_hour': [10],
        'Arrival_Time_minute': [10],
        'Duration_hours': [20],
        'Duration_minutes': [50],
        'Additional_Info': [additional_info]  # Add the 'Additional_Info' column
    })

    # Apply the process_additional_info function to derive the 'Has_Additional_Info' column
    input_data['Has_Additional_Info'] = input_data['Additional_Info'].apply(process_additional_info)

    # Drop the 'Additional_Info' column as it's no longer needed
    input_data = input_data.drop('Additional_Info', axis=1)

    # Ensure the columns are in the correct order
    input_data = input_data[['Airline', 'Source', 'Destination', 'Total_Stops', 'Route_1', 'Route_2', 'Route_3', 'Route_4',
                            'journey_day', 'journey_month', 'journey_year', 'dep_time_hour', 'dep_time_min',
                            'Arrival_Time_hour', 'Arrival_Time_minute', 'Duration_hours', 'Duration_minutes',
                            'Has_Additional_Info']]

    # Predict using the final model
    y_pred = final_model.predict(input_data)

    # Print the predicted value
    print(y_pred)
    # Save the final model to an HDF5 file
    import joblib
    model_path = settings.MEDIA_ROOT + "//" + 'final_model.h5'
    model_filename = model_path
    joblib.dump(final_model, model_filename)



    results = {
        'results_df': results_df.to_html(),
        'grid_search_results': {
            'best_params': best_params,
            'best_score': best_score,
            'test_score': test_score
        },
    }
    return render(request, 'users/ml.html', {'results': results})


    
from django.shortcuts import render
import pandas as pd
import xgboost as xgb
import joblib

def process_additional_info(info):
    if info == 'No info' or info == 'No Info':
        return 'No'
    else:
        return 'Yes'

def extract_duration(duration):
    parts = duration.split()
    hours = 0
    minutes = 0

    # Iterate through each part of the duration
    for part in parts:
        if 'h' in part:
            hours = int(part.replace('h', ''))
        elif 'm' in part:
            minutes = int(part.replace('m', ''))

    return hours, minutes

def predictTrustWorthy(request):
    if request.method == 'POST':
        try:
            # Extracting data from the POST request
            Airline = request.POST.get("Airline")
            Source = request.POST.get("Source")
            Destination = request.POST.get("Destination")
            Total_Stops = request.POST.get("Total_Stops")
            Route_1 = request.POST.get("Route_1")
            Route_2 = request.POST.get("Route_2")
            Route_3 = request.POST.get("Route_3")
            Route_4 = request.POST.get("Route_4")
            journey_day = int(request.POST.get("journey_day"))
            journey_month = int(request.POST.get("journey_month"))
            journey_year = int(request.POST.get("journey_year"))
            dep_time_hour = int(request.POST.get("dep_time_hour"))
            dep_time_min = int(request.POST.get("dep_time_min"))
            Arrival_Time_hour = int(request.POST.get("Arrival_Time_hour"))
            Arrival_Time_minute = int(request.POST.get("Arrival_Time_minute"))
            Duration_hours = int(request.POST.get("Duration_hours"))
            Duration_minutes = int(request.POST.get("Duration_minutes"))
            Additional_Info = 'No info'  # Assuming a default value

            # Create the input_data DataFrame with the user input
            input_data = pd.DataFrame({
                'Airline': [Airline],
                'Source': [Source],
                'Destination': [Destination],
                'Total_Stops': [Total_Stops],
                'Route_1': [Route_1],
                'Route_2': [Route_2],
                'Route_3': [Route_3],
                'Route_4': [Route_4],
                'journey_day': [journey_day],
                'journey_month': [journey_month],
                'journey_year': [journey_year],
                'dep_time_hour': [dep_time_hour],
                'dep_time_min': [dep_time_min],
                'Arrival_Time_hour': [Arrival_Time_hour],
                'Arrival_Time_minute': [Arrival_Time_minute],
                'Duration_hours': [Duration_hours],
                'Duration_minutes': [Duration_minutes],
                'Additional_Info': [Additional_Info]
            })

            # Apply the process_additional_info function to derive the 'Has_Additional_Info' column
            input_data['Has_Additional_Info'] = input_data['Additional_Info'].apply(process_additional_info)

            # Drop the 'Additional_Info' column as it's no longer needed
            input_data = input_data.drop('Additional_Info', axis=1)

            # Ensure the columns are in the correct order
            input_data = input_data[['Airline', 'Source', 'Destination', 'Total_Stops', 'Route_1', 'Route_2', 'Route_3', 'Route_4',
                                    'journey_day', 'journey_month', 'journey_year', 'dep_time_hour', 'dep_time_min',
                                    'Arrival_Time_hour', 'Arrival_Time_minute', 'Duration_hours', 'Duration_minutes',
                                    'Has_Additional_Info']]

            # Load the trained XGBoost regressor model
            model_path = settings.MEDIA_ROOT + "//" + 'final_model.h5'

            xgb_regressor = joblib.load(model_path)

            # Predict using the final model
            y_pred = xgb_regressor.predict(input_data)

            # Assuming you want to return the predicted value
            predicted_value = y_pred[0]

            return render(request, "users/output.html", {'predicted_value': predicted_value})
        except Exception as e:
            return render(request, "users/output.html", {'predicted_value': f'Error: {str(e)}'})
    else:
        return render(request, "users/predictionForm.html")

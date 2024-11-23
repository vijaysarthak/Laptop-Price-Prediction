 ## Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVC

## Load the dataset

data = pd.read_csv('C:/Users/91700/OneDrive/Desktop/SARTHAK/Project UM/Datas/laptop_prices.csv')

## Information Related to Data

data.head()

data.shape

data.describe()

data.info()

data.isnull().sum()

data.corr(numeric_only = True)

data.corr(numeric_only = True)['Price_euros']

sns.heatmap(data.corr(numeric_only = True))

##  Exploratory Data Analysis

data['Company'].value_counts()

# data['Company'].value_counts().plot(kind='bar')

sns.countplot(x='Company', data=data, legend='auto')

# Rotate x-axis labels to be vertical
plt.xticks(rotation=90)

# Show the plot
plt.show()

data['Screen'].value_counts()

sns.countplot(x='Screen', data=data, legend='auto')

# Rotate x-axis labels to be vertical
plt.xticks(rotation=90)

# Show the plot
plt.show()

data['Product'].value_counts()

data['OS'].value_counts()

sns.countplot(x='OS', data=data, legend='auto')

# Rotate x-axis labels to be vertical
plt.xticks(rotation=90)

# Show the plot
plt.show()

data['TypeName'].value_counts()

data['Ram'].value_counts()

sns.countplot(x='Ram', data=data, legend='auto')

# Rotate x-axis labels to be vertical
plt.xticks(rotation=90)

# Show the plot
plt.show()

data['Inches'].value_counts()

sns.countplot(x='Inches', data=data, legend='auto')

# Rotate x-axis labels to be vertical
plt.xticks(rotation=90)

# Show the plot
plt.show()

data['Touchscreen'].value_counts()

sns.countplot(x='Touchscreen', data=data, legend='auto', palette='bright' )

data['IPSpanel'].value_counts()

sns.countplot(x='IPSpanel', data=data, legend='auto', palette='bright')

data['RetinaDisplay'].value_counts()

data['CPU_company'].value_counts()

# Count the occurrences of each CPU_company
CPU_company = data['CPU_company'].value_counts()

# Plot pie chart
plt.pie(CPU_company, labels=CPU_company.index, autopct='%.2f%%')
plt.title("CPU_company")
plt.show()

data['CPU_model'].value_counts()

data['PrimaryStorageType'].value_counts()

# Count the occurrences of each CPU_company
PrimaryStorageType = data['PrimaryStorageType'].value_counts()

# Plot pie chart
plt.pie(PrimaryStorageType, labels=PrimaryStorageType.index, autopct='%.2f%%')
plt.title("PrimaryStorageType")
plt.show()

data['SecondaryStorageType'].value_counts()

sns.countplot(x='SecondaryStorageType', data=data, legend='auto')

# Rotate x-axis labels to be vertical
plt.xticks(rotation=90)

# Show the plot
plt.show()

data['GPU_company'].value_counts()

# Count the occurrences of each CPU_company
GPU_company = data['GPU_company'].value_counts()

# Plot pie chart
plt.pie(GPU_company, labels=GPU_company.index, autopct='%.2f%%')
plt.title("GPU_company")
plt.show()

data['GPU_model'].value_counts()


# Plot boxplot
sns.boxplot(x="OS", y="Price_euros", data=data, palette="bright")
plt.title("Boxplot of Laptop Prices by Operating System")

# Rotate x-axis labels to be vertical
plt.xticks(rotation=90)

# Show the plot
plt.show()

# Plot barplot
sns.barplot(x = 'Touchscreen', y = 'Price_euros', hue ='Screen', palette="bright", data = data)
plt.title("Barplot of Laptop Prices by Touch Screen")

# Rotate x-axis labels to be vertical
plt.xticks(rotation=90)

# Show the plot
plt.show()

# Plot boxplot
sns.barplot(x = 'Touchscreen', y = 'Price_euros', palette="bright", data = data)
plt.title("Barplot of Laptop Prices by Touch Screen")

# Rotate x-axis labels to be vertical
plt.xticks(rotation=90)

# Show the plot
plt.show()

# Plot boxplot
sns.barplot(x="PrimaryStorage", y="Price_euros", data=data, palette="bright", hue = 'SecondaryStorage')
plt.title("Barplot of Laptop Prices by PrimaryStorage")

# Rotate x-axis labels to be vertical
plt.xticks(rotation=90)

# Show the plot
plt.show()

# Plot boxplot
sns.barplot(x="RetinaDisplay", y="Price_euros", data=data, palette="bright", hue = 'PrimaryStorage')
plt.title("Barplot of Laptop Prices by Retina Display")

# Rotate x-axis labels to be vertical
plt.xticks(rotation=90)

# Show the plot
plt.show()

sns.scatterplot(data = data , x= 'CPU_freq', y = 'Ram', hue = 'OS')

sns.scatterplot(data = data , x= np.log(data['Weight']), y = np.log(data['Price_euros']), hue = 'OS')

sns.displot(data['Price_euros'])

sns.displot(x= np.log(data['Price_euros']))

# sns.pairplot(data)

## Define features and target

X = data.drop('Price_euros', axis=1)
y = np.log(data['Price_euros'])

## Identify categorical and numerical columns

categorical_cols = X.select_dtypes(include=['object']).columns
categorical_cols

numerical_cols = X.select_dtypes(include=['float64', 'int64', 'int32']).columns
numerical_cols

## Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Encoding for categorical features and Preprocessing pipelines

preprocessor = ColumnTransformer(
    transformers=[        
        ('cat', OneHotEncoder(drop = 'first', handle_unknown= 'ignore'), ['Company', 'Product', 'TypeName', 'OS', 'Screen', 'Touchscreen', 'IPSpanel', 'RetinaDisplay', 'CPU_company', 'CPU_model', 'PrimaryStorageType','SecondaryStorageType', 'GPU_company', 'GPU_model']),
        ('num', StandardScaler(), ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage'])
    ], remainder = 'passthrough')

regressor = LinearRegression()

pipe_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

# Train the Model
pipe_reg.fit(X_train, y_train)

y_pred_reg = pipe_reg.predict(X_test)

# Evaluate the test model
mse_test_reg = mean_squared_error(y_test, y_pred_reg)
r2_test_reg = r2_score(y_test, y_pred_reg)

print(f"Mean Squared Error Test data: {mse_test_reg}")
print(f"R-squared Test data: {r2_test_reg}")

## Ridge Regression

rid_prepro = ColumnTransformer(
    transformers=[        
        ('cat', OneHotEncoder(drop = 'first', handle_unknown= 'ignore'), ['Company', 'Product', 'TypeName', 'OS', 'Screen', 'Touchscreen', 'IPSpanel', 'RetinaDisplay', 'CPU_company', 'CPU_model', 'PrimaryStorageType','SecondaryStorageType', 'GPU_company', 'GPU_model']),
        ('num', StandardScaler(), ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage'])
    ], remainder = 'passthrough')
    
rid_reg = Ridge(alpha=0.9)

pipe_rid = Pipeline([
    ('rid_prepro', rid_prepro),
    ('rid_reg', rid_reg)
])

# Train the Model
pipe_rid.fit(X_train, y_train)

y_pred_rid = pipe_rid.predict(X_test)

# Evaluate the test model
mse_test_rid = mean_squared_error(y_test, y_pred_rid)
r2_test_rid = r2_score(y_test, y_pred_rid)

print(f"Mean Squared Error Test data: {mse_test_rid}")
print(f"R-squared Test data: {r2_test_rid}")

# Visualize test results
sns.jointplot(x = y_test, y= y_pred_rid, kind="reg")
plt.xlabel("Actual Prices of Test data")
plt.ylabel("Predicted Prices of Test data")
plt.show()

## Lasso Regression

las_prepro = ColumnTransformer(
    transformers=[        
        ('cat', OneHotEncoder(drop = 'first', handle_unknown= 'ignore'), ['Company', 'Product', 'TypeName', 'OS', 'Screen', 'Touchscreen', 'IPSpanel', 'RetinaDisplay', 'CPU_company', 'CPU_model', 'PrimaryStorageType','SecondaryStorageType', 'GPU_company', 'GPU_model']),
        ('num', StandardScaler(), ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage'])
    ], remainder = 'passthrough')

las_reg = Lasso(alpha=0.0001)

pipe_las = Pipeline([
    ('las_prepro', las_prepro),
    ('las_reg', las_reg)
])

# Train the Model
pipe_las.fit(X_train, y_train)

y_pred_las = pipe_las.predict(X_test)

# Evaluate the test model
mse_test_las = mean_squared_error(y_test, y_pred_las)
r2_test_las = r2_score(y_test, y_pred_las)

print(f"Mean Squared Error Test data: {mse_test_las}")
print(f"R-squared Test data: {r2_test_las}")

##  KNeighbors Regressor

knn_prepro = ColumnTransformer(
    transformers=[        
        ('cat', OneHotEncoder(drop = 'first', handle_unknown= 'ignore'), ['Company', 'Product', 'TypeName', 'OS', 'Screen', 'Touchscreen', 'IPSpanel', 'RetinaDisplay', 'CPU_company', 'CPU_model', 'PrimaryStorageType','SecondaryStorageType', 'GPU_company', 'GPU_model']),
        ('num', StandardScaler(), ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage'])
    ], remainder = 'passthrough')

knn = KNeighborsRegressor(n_neighbors=5)

pipe_knn = Pipeline([
    ('knn_prepro', knn_prepro),
    ('knn', knn)
])

# Train the Model
pipe_knn.fit(X_train, y_train)

y_pred_knn = pipe_knn.predict(X_test)

# Evaluate the test model
mse_test_knn = mean_squared_error(y_test, y_pred_knn)
r2_test_knn = r2_score(y_test, y_pred_knn)

print(f"Mean Squared Error Test data: {mse_test_knn}")
print(f"R-squared Test data: {r2_test_knn}")

## Decision Tree Regressor

dtr_prepro = ColumnTransformer(
    transformers=[        
        ('cat', OneHotEncoder(drop = 'first', handle_unknown= 'ignore'), ['Company', 'Product', 'TypeName', 'OS', 'Screen', 'Touchscreen', 'IPSpanel', 'RetinaDisplay', 'CPU_company', 'CPU_model', 'PrimaryStorageType','SecondaryStorageType', 'GPU_company', 'GPU_model']),
        ('num', StandardScaler(), ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage'])
    ], remainder = 'passthrough')

dtr = DecisionTreeRegressor(max_depth=12, min_samples_split=12, max_features=0.8, random_state=5)

pipe_dtr = Pipeline([
    ('dtr_prepro', dtr_prepro),
    ('dtr', dtr)
])

# Train the Model
pipe_dtr.fit(X_train, y_train)

y_pred_dtr = pipe_dtr.predict(X_test)

# Evaluate the test model
mse_test_dtr = mean_squared_error(y_test, y_pred_dtr)
r2_test_dtr = r2_score(y_test, y_pred_dtr)

print(f"Mean Squared Error Test data: {mse_test_dtr}")
print(f"R-squared Test data: {r2_test_dtr}")

## Random Forest Regressor

rfr_prepro = ColumnTransformer(
    transformers=[        
        ('cat', OneHotEncoder(drop = 'first', handle_unknown= 'ignore'), ['Company', 'Product', 'TypeName', 'OS', 'Screen', 'Touchscreen', 'IPSpanel', 'RetinaDisplay', 'CPU_company', 'CPU_model', 'PrimaryStorageType','SecondaryStorageType', 'GPU_company', 'GPU_model']),
        ('num', StandardScaler(), ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage'])
    ], remainder = 'passthrough')

rfr = RandomForestRegressor(n_estimators = 100 ,max_features=0.5, max_depth=15, max_samples=0.7, random_state=3)

pipe_rfr = Pipeline([
    ('rfr_prepro', rfr_prepro),
    ('rfr', rfr)
])

# Train the Model
pipe_rfr.fit(X_train, y_train)

y_pred_rfr = pipe_rfr.predict(X_test)

# Evaluate the test model
mse_test_rfr = mean_squared_error(y_test, y_pred_rfr)
r2_test_rfr = r2_score(y_test, y_pred_rfr)

print(f"Mean Squared Error Test data: {mse_test_rfr}")
print(f"R-squared Test data: {r2_test_rfr}")

## AdaBoost Regressor

ab_prepro = ColumnTransformer(
    transformers=[        
        ('cat', OneHotEncoder(drop = 'first', handle_unknown= 'ignore'), ['Company', 'Product', 'TypeName', 'OS', 'Screen', 'Touchscreen', 'IPSpanel', 'RetinaDisplay', 'CPU_company', 'CPU_model', 'PrimaryStorageType','SecondaryStorageType', 'GPU_company', 'GPU_model']),
        ('num', StandardScaler(), ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage'])
    ], remainder = 'passthrough')

ab_regg = AdaBoostRegressor(n_estimators =500, learning_rate=0.9, random_state=3)

pipe_ab = Pipeline([
    ('ab_prepro', ab_prepro),
    ('ab_regg', ab_regg)
])

# Train the Model
pipe_ab.fit(X_train, y_train)

y_pred_ab = pipe_ab.predict(X_test)

# Evaluate the test model
mse_test_ab = mean_squared_error(y_test, y_pred_ab)
r2_test_ab = r2_score(y_test, y_pred_ab)

print(f"Mean Squared Error Test data: {mse_test_ab}")
print(f"R-squared Test data: {r2_test_ab}")

## Gradient Boosting Regressor

gbr_prepro = ColumnTransformer(
    transformers=[        
        ('cat', OneHotEncoder(drop = 'first', handle_unknown= 'ignore'), ['Company', 'Product', 'TypeName', 'OS', 'Screen', 'Touchscreen', 'IPSpanel', 'RetinaDisplay', 'CPU_company', 'CPU_model', 'PrimaryStorageType','SecondaryStorageType', 'GPU_company', 'GPU_model']),
        ('num', StandardScaler(), ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage'])
    ], remainder = 'passthrough')

gb_regg = GradientBoostingRegressor(n_estimators = 1000, random_state=3, alpha=0.1)

pipe_gb = Pipeline([
    ('gbr_prepro', gbr_prepro),
    ('gb_regg', gb_regg)
])

# Train the Model
pipe_gb.fit(X_train, y_train)

y_pred_gb = pipe_gb.predict(X_test)

# Evaluate the test model
mse_test_gb = mean_squared_error(y_test, y_pred_gb)
r2_test_gb = r2_score(y_test, y_pred_gb)

print(f"Mean Squared Error Test data: {mse_test_gb}")
print(f"R-squared Test data: {r2_test_gb}")

## XGBoost

xgb_prepro = ColumnTransformer(
    transformers=[        
        ('cat', OneHotEncoder(drop = 'first', handle_unknown= 'ignore'), ['Company', 'Product', 'TypeName', 'OS', 'Screen', 'Touchscreen', 'IPSpanel', 'RetinaDisplay', 'CPU_company', 'CPU_model', 'PrimaryStorageType','SecondaryStorageType', 'GPU_company', 'GPU_model']),
        ('num', StandardScaler(), ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage'])
    ], remainder = 'passthrough')

xgb_regg = XGBRegressor(n_estimators = 100, learning_rate=0.9, max_depth=10, random_state=3)

pipe_xgb = Pipeline([
    ('xgb_prepro', xgb_prepro),
    ('xgb_regg', xgb_regg)
])

# Train the Model
pipe_xgb.fit(X_train, y_train)

y_pred_xgb = pipe_xgb.predict(X_test)

# Evaluate the test model
mse_test_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_test_xgb = r2_score(y_test, y_pred_xgb)

print(f"Mean Squared Error Test data: {mse_test_xgb}")
print(f"R-squared Test data: {r2_test_xgb}")

## Exporting the Model

import pickle
pickle.dump(data,open('data.pkl','wb'))
pickle.dump(pipe_rid,open('pipe.pkl','wb'))
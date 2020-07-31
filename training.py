#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#%%
df = pd.read_csv("./Car-Price-Prediction/car data.csv")


#%%
df.head()


#%%
df.shape


#%%
print(df.Car_Name.unique())
print(df.Seller_Type.unique())
print(df.Fuel_Type.unique())
print(df.Transmission.unique())
print(df.Owner.unique())


#%%
df.isnull().sum()


#%%
df['no_year'] = 2020 - df['Year']
df.drop(["Year", "Car_Name"], axis=1, inplace=True)


#%%
df.head(3)


#%%
df = pd.get_dummies(df,drop_first=True)


#%%
df.head(3)


#%%



#%%
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)


#%%
# Independent and dependent feature
X=df.iloc[:,1:]
y=df.iloc[:,0]


#%%
X.head(3)


#%%
y.head(3)


#%%
# Feature Importance
from sklearn.ensemble import ExtraTreesRegressor


#%%
model=ExtraTreesRegressor()
model.fit(X,y)


#%%
print(model.feature_importances_)


#%%
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(5).plot(kind='barh')
plt.show()


#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


#%%
X_train.shape


#%%
from sklearn.ensemble import RandomForestRegressor


#%%



#%%
#Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


#%%
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


#%%
rf = RandomForestRegressor()


#%%
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


#%%
rf_random.fit(X_train, y_train)


#%%
rf_random.best_params_


#%%
rf_random.best_score_


#%%
predictions=rf_random.predict(X_test)


#%%
sns.distplot(y_test-predictions)


#%%
plt.scatter(y_test,predictions)


#%%
from sklearn import metrics


#%%
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


#%%
import pickle
# open a file, where you ant to store the data
file = open('my_pickel.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


#%%



#%%



#%%



#%%



#%%



#%%




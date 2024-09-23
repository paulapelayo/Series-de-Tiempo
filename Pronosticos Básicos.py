#!/usr/bin/env python
# coding: utf-8

# In[13]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


# In[14]:


gold_symbol = 'GC=F'


# In[15]:


start_date = '2020-01-01'
end_date = '2024-08-23'


# In[16]:


gold_data = yf.download(gold_symbol, start=start_date, end=end_date)


# In[17]:


print(gold_data.head())


# In[18]:


pip install statsforecast


# In[22]:


from statsforecast.models import HistoricAverage
y_mean = gold_symbol
model = HistoricAverage()
model = model.fit(y=gold_symbol)
y_hat_dict = model.predict(h=3)

y_hat_dict


# In[ ]:


pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])


# In[ ]:


# Specify the quarters predicted:
months_pred = pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])

# Create a dataframe with the quarters and values predicted:
Y_pred_df = pd.DataFrame({'mean_forecast':y_hat_dict["mean"]}, index = months_pred)
     


# In[ ]:


pd.concat([data, Y_pred_df])


# In[ ]:


frame = pd.concat([data, Y_pred_df])
frame.columns = ['Values', 'mean_forecast']


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize = (20, 8))

# Concatenate the dataframe of predicted values with the dataframe of observed values:
plot_df = pd.concat([data, Y_pred_df])
plot_df.columns = ['Values', 'mean_forecast']
plot_df[['Values', 'mean_forecast']].plot(ax=ax, linewidth=2)

# Specify graph features:
ax.set_title('remesas', fontsize=22)
ax.set_ylabel('remesas', fontsize=20)
ax.set_xlabel('mes', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()


# In[ ]:



fig, ax = plt.subplots(1, 1, figsize = (20, 8))

# Concatenate the dataframe of predicted values with the dataframe of observed values:
plot_df = pd.concat([data, Y_pred_df])
plot_df.columns = ['Values', 'mean_forecast']
plot_df[-24:].plot(ax=ax, linewidth=2)

# Specify graph features:
ax.set_title('remesas', fontsize=22)
ax.set_ylabel('remesas', fontsize=20)
ax.set_xlabel('mes', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()


# # Naive Method

# In[ ]:


# Naive's usage example:
from statsforecast.models import Naive

# Define the model, fit and predict:
model = Naive()
model = model.fit(y=data.values)
y_hat_dict = model.predict(h=3)

y_hat_dict


# In[ ]:


# Create a column with the values predicted:
Y_pred_df["naive_forecast"] = y_hat_dict["mean"]


# In[ ]:


Y_pred_df


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize = (20, 8))

# Concatenate the dataframe of predicted values with the dataframe of observed values:
plot_df = pd.concat([data, Y_pred_df])
plot_df.columns = ['Values', 'mean_forecast', 'naive_forecast']
plot_df[-24:].plot(ax=ax, linewidth=2)

# Specify graph features:
ax.set_title('remesas', fontsize=22)
ax.set_ylabel('remesas', fontsize=20)
ax.set_xlabel('mes', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()


# # Seasonal Naive

# In[ ]:


# SeasonalNaive's usage example:
from statsforecast.models import SeasonalNaive

# Define the model, fit and predict:
model = SeasonalNaive(season_length=12)
model = model.fit(y=data.values)
y_hat_dict = model.predict(h=3)

y_hat_dict


# In[ ]:


# Create a column with the values predicted:
Y_pred_df["seasonal_naive_forecast"] = y_hat_dict["mean"]


# In[ ]:



fig, ax = plt.subplots(1, 1, figsize = (20, 8))

# Concatenate the dataframe of predicted values with the dataframe of observed values:
plot_df = pd.concat([data, Y_pred_df])
plot_df.columns = ['Values', 'mean_forecast', 'naive_forecast', 'seasonal_naive_forecast']
plot_df[-24:].plot(ax=ax, linewidth=2)

# Specify graph features:
ax.set_title('remesas', fontsize=22)
ax.set_ylabel('remesas', fontsize=20)
ax.set_xlabel('mes', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()


# In[ ]:


# RandomWalkWithDrift's usage example:
from statsforecast.models import RandomWalkWithDrift

# Define the model, fit and predict:
model = RandomWalkWithDrift()
model = model.fit(y=data.values)
y_hat_dict = model.predict(h=3)

y_hat_dict


# In[ ]:


# Create a column with the values predicted:
Y_pred_df["drift_forecast"] = y_hat_dict["mean"]


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize = (20, 8))

# Concatenate the dataframe of predicted values with the dataframe of observed values:
plot_df = pd.concat([data, Y_pred_df])
plot_df.columns = ['Values', 'mean_forecast', 'naive_forecast', 'seasonal_naive_forecast', 'drift_forecast']
plot_df[-24:].plot(ax=ax, linewidth=2)

# Specify graph features:
ax.set_title('remesas', fontsize=22)
ax.set_ylabel('remesas', fontsize=20)
ax.set_xlabel('mes', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()


# # Drift Method

# In[ ]:


#Import the libraries that we are going to use for the analysis:
import pandas as pd
import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import __all__
from statsforecast.models import HistoricAverage


# In[ ]:


# Create a dataframe from a csv file:
#"/content/drive/MyDrive/series_tiempo/Assets/remesas banxico.xlsx"
df = pd.read_csv("/content/drive/MyDrive/series_tiempo/Assets/aus-production.csv", sep=";")

# Create a dataframe with beer production:
beer = df[["Quarter","Beer"]]

#Inferior limit:
beer_mask=beer['Quarter']>="1992 Q1"
filtered_beer = beer[beer_mask]

#Superior limit:
beer_mask=filtered_beer['Quarter']<="2006 Q4"
beer = filtered_beer[beer_mask]

# Create an array with the observed values
y_beer = beer["Beer"].values

# Mean method:
model = HistoricAverage()
model = model.fit(y=y_beer)
mean = model.predict(h=14)
     


# In[ ]:


'''
The function augment() return a DataFrame with five columns (model, time_var, obs_values, fitted_values and residuals) and has 3 arguments:

    * model_name: String variable. It´s reffer to what kind of model are we using (Mean, Naive, SNaive, ETS, etc.).

    * time_var: Column of the input dataframe or numpy array. It´s reffer to the time index of the observed values.

    * obs_values: Column of the input dataframe or numpy array. It´s reffer to the the observed values.

'''

def augment(model_name, time_var, obs_values):

    type_model_list = []
    for n in range(len(obs_values)):
        type_model_list.append(model_name)
        n += 1
    fitted_values = model.predict_in_sample()
    residuals = obs_values - fitted_values["fitted"]

    augment_df = pd.DataFrame({'model':type_model_list,
                        'time_var':time_var,
                        'obs_values':obs_values,
                        'fitted_values':fitted_values["fitted"],
                        'residuals':residuals})

    return(augment_df)
     


# In[ ]:


# augment(model_name, time_var, obs_values)
augment_df = augment("Mean",beer["Quarter"],beer["Beer"])

augment_df.tail()


# # Valores Ajustados y Residuales

# In[ ]:


# Define the model, fit and predict:
model = SeasonalNaive(season_length=12)
model = model.fit(y=data.values)
mean = model.predict(h=12)


# In[ ]:


# augment(model_name, time_var, obs_values)
augment_df = augment("Mean",data.index,data)


# In[ ]:


augment_df


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize = (20, 8))
augment_df['residuals'].plot(ax=ax, linewidth=2)

# Specify graph features:
ax.set_title('Residuales del modelo ajustado Remesas Banxico 1995 - 2023', fontsize=22)
ax.set_ylabel('Miles de millones de dolares', fontsize=20)
ax.set_xlabel('Mes', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()


# In[ ]:


# Creating histogram
fig, axs = plt.subplots(1, 1,
                        figsize =(20, 8),
                        tight_layout = True)

axs.hist(augment_df["residuals"], bins = 20)

# Specify graph features:
axs.set_title('histograma residuales', fontsize=22)
axs.set_ylabel('conteo', fontsize=20)
axs.set_xlabel('residuales', fontsize=20)

# Show plot
plt.show()


# In[ ]:


import math

ticker_data = augment_df["residuals"]
ticker_data_acf = [ticker_data.autocorr(i) for i in range(1,25)]

test_df = pd.DataFrame([ticker_data_acf]).T
test_df.columns = ['Autocorr']
test_df.index += 1
test_df.plot(kind='bar', width = 0.05, figsize = (20, 4))

# Statisfical significance.
n = len(augment_df['residuals'])
plt.axhline(y = 2/math.sqrt(n), color = 'r', linestyle = 'dashed')
plt.axhline(y = -2/math.sqrt(n), color = 'r', linestyle = 'dashed')

# Adding plot title.
plt.title("Residuals from the Naive method")

# Providing x-axis name.
plt.xlabel("lag[1]")

# Providing y-axis name.
plt.ylabel("ACF")


# # Residual diagnostics

# In[ ]:


import statsmodels


# In[ ]:


ljung_box = statsmodels.stats.diagnostic.acorr_ljungbox(test_df, lags=12, model_df=0)
ljung_box.tail(1)


# In[ ]:


fig = statsmodels.api.qqplot(augment_df[['residuals']].values, line='q')
plt.show()


# In[ ]:


from scipy.stats import lognorm
np.random.seed(1)

#generate dataset that contains 1000 log-normal distributed values
lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)

#create Q-Q plot with 45-degree line added to plot
fig = statsmodels.api.qqplot(lognorm_dataset, line='q')

plt.show()


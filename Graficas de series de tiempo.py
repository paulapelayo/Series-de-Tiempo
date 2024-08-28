#!/usr/bin/env python
# coding: utf-8

# # Grafico del Oro

# In[17]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

gold_symbol = 'GC=F'  # Contrato de futuros del oro

# Rango de fechas
start_date = '2020-01-01'
end_date = '2024-08-23'

# Descarga de datos
gold_data = yf.download(gold_symbol, start=start_date, end=end_date)

#Primeros 5 registros
print(gold_data)

# Grafico
plt.figure(figsize=(12,6))
plt.plot(gold_data['Close'], label='Precio de cierre')
plt.title('Precio del Oro en 2022')
plt.xlabel('Fecha')
plt.ylabel('Precio en USD')
plt.legend()
plt.show()


# Se puede observar una tendencia de alza pero al mismo tiempo los datos han tenido un par de ocasiones en la que el precio del oro baja. 

# In[21]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot


# In[20]:


month_plot(gold_data, ylabel='Oro')


# In[18]:


from pandas.plotting import lag_plot
lag_plot(gold_data, lag=1)


# In[23]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
gold_data = yf.download(gold_symbol, start=start_date, end=end_date)
plot_acf(gold_data, lags=24)


# In[ ]:





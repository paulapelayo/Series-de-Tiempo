#!/usr/bin/env python
# coding: utf-8

# # Regresión Lineal

# # mi variable x es el dolar

# In[1]:


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


# In[2]:


pip install yfinance


# In[6]:


import yfinance as yf
import matplotlib.pyplot as plt

data = yf.download('USDMXN=X', start='2020-01-01', end='2024-08-31')

plt.figure(figsize=(10,6))
plt.plot(data.index, data['Close'], label='Precio USDMXN')
plt.title('Precio del Dólar (USDMXN) desde enero 2020 hasta agosto 2024')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre (USDMXN)')
plt.grid(True)
plt.legend()
plt.show()


# In[7]:


data = data["2003":]


# In[ ]:


usdmxn = USDMXN["2004":]


# In[13]:


import yfinance as yf
import pandas as pd

data = yf.download('USDMXN=X', start='2020-01-01', end='2024-08-31')

dates = ['2020-01-01', '2021-01-01', '2022-01-03', '2023-01-02', '2024-01-01']  # Ajustes por días feriados

resultados = []

for date in dates:
    row = data.loc[data.index == date]
    
    if not row.empty:
        price = row['Close'].values[0]
        resultados.append([date, f"{price:.4f} MXN"])
    else:
        resultados.append([date, 'Datos no disponibles'])

tabla = pd.DataFrame(resultados, columns=['Fecha', 'Precio USD/MXN'])

print(tabla)


# In[14]:


import yfinance as yf
import pandas as pd

data = yf.download('USDMXN=X', start='2020-01-01', end='2024-08-31')

data.reset_index(inplace=True)

data['mes'] = data['Date'].dt.month

data = pd.get_dummies(data, columns=['mes'], prefix="", prefix_sep="", drop_first=True, dtype=float)

print(data.head())


# In[15]:


import yfinance as yf
import pandas as pd

data = yf.download('USDMXN=X', start='2020-01-01', end='2024-08-31')

Q1 = data['Close'].quantile(0.25)
Q3 = data['Close'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['Close'] < lower_bound) | (data['Close'] > upper_bound)]

print(f"Outliers encontrados:\n{outliers[['Close']]}")


# Escogere el mas alto que fue el 2020-04-27 que el precio del dolar estaba a 24.98

# In[22]:


print (24.98)


# In[25]:


start = pd.Series(data = [1], index=pd.to_datetime(["2020-01-01"]), name='start_outlier')
end = pd.Series(data = [1], index=pd.to_datetime(["2024-08-31"]), name='end_outlier')
     


# In[26]:


start


# In[27]:


import yfinance as yf
import pandas as pd

data = yf.download('USDMXN=X', start='2020-01-01', end='2024-08-31')

data['lag1'] = data['Close'].shift(1)
data['lag2'] = data['Close'].shift(2)
data['lag3'] = data['Close'].shift(3)
data['lag4'] = data['Close'].shift(4)
data['lag5'] = data['Close'].shift(5)
data['lag6'] = data['Close'].shift(6)
data['lag7'] = data['Close'].shift(7)
data['lag8'] = data['Close'].shift(8)
data['lag9'] = data['Close'].shift(9)
data['lag10'] = data['Close'].shift(10)
data['lag11'] = data['Close'].shift(11)
data['lag12'] = data['Close'].shift(12)

print(data[['Close', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8', 'lag9', 'lag10', 'lag11', 'lag12']].head(15))


# # Modelo de regresión

# In[31]:


import yfinance as yf
import pandas as pd
import statsmodels.api as sm

data = yf.download('USDMXN=X', start='2020-01-01', end='2024-08-31')

for i in range(1, 13):
    data[f'lag{i}'] = data['Close'].shift(i)

data['Y'] = data['Close']

data = data.dropna()

X = data.drop(columns=['Y', 'Close'])

X = sm.add_constant(X)

model = sm.OLS(data['Y'], X).fit()

print(model.summary())


# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar los datos
data = pd.read_csv('tipo_de_cambio.csv')  # Asegúrate de que tu archivo CSV tenga las columnas adecuadas
data['Fecha'] = pd.to_datetime(data['Fecha'])
data['Tiempo'] = (data['Fecha'] - data['Fecha'].min()).dt.days  # Tiempo en días desde la primera fecha

# Definir variables
X = data['Tiempo'].values.reshape(-1, 1)  # Variable independiente
y = data['Tipo_de_cambio'].values  # Variable dependiente

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Hacer predicciones
predicciones = modelo.predict(X_test)

# Visualizar los resultados
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X_test, predicciones, color='red', label='


# In[40]:


from sklearn.linear_model import LinearRegression


# In[41]:


LinearRegression().fit(data.drop(columns=['Y']), data['Y']).score(data.drop(columns=['Y']), data['Y'])


# In[44]:


import scipy as sp


# In[45]:


sp.stats.boxcox(data['Y'])[0].shape


# De estos modelos de regresión linealque vimos en clase creo que el más conveniente para mi variable seria el de LAGS porque el precio y cambio del oro esta en constante cambio y este cambia de manera diaria pero al igual es muy importante notar yo mar en cuenta todas estas variables. 

# 

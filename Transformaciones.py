#!/usr/bin/env python
# coding: utf-8

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


# In[6]:


import yfinance as yf
import pandas as pd
import numpy as np

# Símbolo del contrato de futuros del oro
gold_symbol = 'GC=F'

# Rango de fechas
start_date = '2020-01-01'
end_date = '2024-08-23'

# Descarga de datos del oro
gold_data = yf.download(gold_symbol, start=start_date, end=end_date)

# Mostrar los primeros 5 registros
print(gold_data.head())

# Calcular los retornos logarítmicos del precio de cierre
gold_data['Return'] = np.log(gold_data['Close'] / gold_data['Close'].shift(1))

# Eliminar los valores nulos (resultado del shift)
gold_data.dropna(inplace=True)

# Calcular la varianza de los retornos
variance = gold_data['Return'].var()

print(f"Varianza de los retornos: {variance}")


# In[8]:


plt.figure(figsize=(10, 6))
plt.plot(gold_data.index, gold_data['Return'], label='Retorno del oro', color='blue')
plt.title('Retornos Logarítmicos del Oro')
plt.xlabel('Fecha')
plt.ylabel('Retorno Logarítmico')
plt.grid(True)
plt.legend()
plt.show()


# # Elimina valores antiguos que no representan la actualidad
# 

# In[11]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Símbolo del contrato de futuros del oro
gold_symbol = 'GC=F'

# Rango de fechas
start_date = '2020-01-01'
end_date = '2024-08-23'

# Descarga de datos del oro
gold_data = yf.download(gold_symbol, start=start_date, end=end_date)

# Mostrar los primeros 5 registros
print(gold_data.head())

# Calcular los retornos logarítmicos del precio de cierre
gold_data['Return'] = np.log(gold_data['Close'] / gold_data['Close'].shift(1))

# Eliminar los valores nulos (resultado del shift)
gold_data.dropna(inplace=True)

# Eliminar datos antiguos (antes de 2022)
recent_data = gold_data[gold_data.index >= '2022-01-01']

# Mostrar los primeros 5 registros de los datos filtrados
print(recent_data.head())

# Calcular la varianza de los retornos con datos recientes
variance = recent_data['Return'].var()

print(f"Varianza de los retornos recientes: {variance}")

# Graficar el retorno de los precios de cierre recientes
plt.figure(figsize=(10, 6))
plt.plot(recent_data.index, recent_data['Return'], label='Retorno del oro (recientes)', color='blue')
plt.title('Retornos Logarítmicos del Oro (Datos Recientes)')
plt.xlabel('Fecha')
plt.ylabel('Retorno Logarítmico')
plt.grid(True)
plt.legend()
plt.show()


# # Diferenciación

# In[12]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Símbolo del contrato de futuros del oro
gold_symbol = 'GC=F'

# Rango de fechas
start_date = '2020-01-01'
end_date = '2024-08-23'

# Descarga de datos del oro
gold_data = yf.download(gold_symbol, start=start_date, end=end_date)

# Mostrar los primeros 5 registros
print(gold_data.head())

# Aplicar diferenciación en el precio de cierre
gold_data['Diff_Close'] = gold_data['Close'].diff()

# Eliminar los valores nulos (resultado de la diferenciación)
gold_data.dropna(inplace=True)

# Mostrar los primeros 5 registros diferenciados
print(gold_data[['Close', 'Diff_Close']].head())

# Graficar el precio de cierre original y su diferenciación
plt.figure(figsize=(10, 6))

# Gráfico del precio original
plt.subplot(2, 1, 1)
plt.plot(gold_data.index, gold_data['Close'], label='Precio de Cierre', color='blue')
plt.title('Precio de Cierre del Oro')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.grid(True)

# Gráfico de la serie diferenciada
plt.subplot(2, 1, 2)
plt.plot(gold_data.index, gold_data['Diff_Close'], label='Precio Diferenciado', color='red')
plt.title('Diferenciación del Precio de Cierre del Oro')
plt.xlabel('Fecha')
plt.ylabel('Diferencia de Precio')
plt.grid(True)

plt.tight_layout()
plt.show()


# # Retorno

# In[13]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Símbolo del contrato de futuros del oro
gold_symbol = 'GC=F'

# Rango de fechas
start_date = '2020-01-01'
end_date = '2024-08-23'

# Descarga de datos del oro
gold_data = yf.download(gold_symbol, start=start_date, end=end_date)

# Mostrar los primeros 5 registros
print(gold_data.head())

# Calcular los retornos de inversión (retornos logarítmicos)
gold_data['Return'] = np.log(gold_data['Close'] / gold_data['Close'].shift(1))

# Eliminar los valores nulos (resultado del shift)
gold_data.dropna(inplace=True)

# Calcular el rendimiento acumulado
gold_data['Cumulative_Return'] = (1 + gold_data['Return']).cumprod() - 1

# Mostrar los primeros 5 registros de retornos
print(gold_data[['Close', 'Return', 'Cumulative_Return']].head())

# Graficar los retornos de inversión
plt.figure(figsize=(12, 6))
plt.plot(gold_data.index, gold_data['Cumulative_Return'], label='Retorno Acumulado del Oro', color='gold')
plt.title('Retorno Acumulado de Inversión en Oro')
plt.xlabel('Fecha')
plt.ylabel('Retorno Acumulado')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Línea horizontal en 0
plt.grid(True)
plt.legend()
plt.show()


# # Escalamiento y desplazamiento

# In[14]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Símbolo del contrato de futuros del oro
gold_symbol = 'GC=F'

# Rango de fechas
start_date = '2020-01-01'
end_date = '2024-08-23'

# Descarga de datos del oro
gold_data = yf.download(gold_symbol, start=start_date, end=end_date)

# Calcular los retornos logarítmicos del precio de cierre
gold_data['Return'] = np.log(gold_data['Close'] / gold_data['Close'].shift(1))

# Eliminar los valores nulos (resultado del shift)
gold_data.dropna(inplace=True)

# Escalamiento (normalización Min-Max)
scaler_min_max = MinMaxScaler()
gold_data['Scaled_Return'] = scaler_min_max.fit_transform(gold_data[['Return']])

# Desplazamiento (estandarización z-score)
scaler_standard = StandardScaler()
gold_data['Standardized_Return'] = scaler_standard.fit_transform(gold_data[['Return']])

# Mostrar los primeros 5 registros
print(gold_data[['Return', 'Scaled_Return', 'Standardized_Return']].head())

# Graficar los retornos originales, escalados y estandarizados
plt.figure(figsize=(14, 8))

# Gráfico de retornos originales
plt.subplot(3, 1, 1)
plt.plot(gold_data.index, gold_data['Return'], label='Retornos Originales', color='blue')
plt.title('Retornos Logarítmicos del Oro')
plt.ylabel('Retorno')
plt.grid(True)

# Gráfico de retornos escalados
plt.subplot(3, 1, 2)
plt.plot(gold_data.index, gold_data['Scaled_Return'], label='Retornos Escalados (Min-Max)', color='orange')
plt.title('Retornos Escalados (Min-Max)')
plt.ylabel('Retorno Escalado')
plt.grid(True)

# Gráfico de retornos estandarizados
plt.subplot(3, 1, 3)
plt.plot(gold_data.index, gold_data['Standardized_Return'], label='Retornos Estandarizados (Z-Score)', color='green')
plt.title('Retornos Estandarizados (Z-Score)')
plt.ylabel('Retorno Estandarizado')
plt.grid(True)

plt.tight_layout()
plt.show()


# # Desplazamiento

# In[ ]:


# import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Símbolo del contrato de futuros del oro
gold_symbol = 'GC=F'

# Rango de fechas
start_date = '2020-01-01'
end_date = '2024-08-23'

# Descarga de datos del oro
gold_data = yf.download(gold_symbol, start=start_date, end=end_date)

# Calcular los retornos logarítmicos del precio de cierre
gold_data['Return'] = np.log(gold_data['Close'] / gold_data['Close'].shift(1))

# Eliminar los valores nulos (resultado del shift)
gold_data.dropna(inplace=True)

# Desplazamiento de los retornos (shift)
# Cambiar el número en shift() para desplazar por más días
gold_data['Return_Lag_1'] = gold_data['Return'].shift(1)
gold_data['Return_Lag_2'] = gold_data['Return'].shift(2)

# Mostrar los primeros 5 registros
print(gold_data[['Return', 'Return_Lag_1', 'Return_Lag_2']].head())

# Graficar los retornos y sus desplazamientos
plt.figure(figsize=(12, 6))

# Gráfico de retornos originales
plt.plot(gold_data.index, gold_data['Return'], label='Retornos Originales', color='blue')
plt.plot(gold_data.index, gold_data['Return_Lag_1'], label='Retorno Lag 1', linestyle='--', color='orange')
plt.plot(gold_data.index, gold_data['Return_Lag_2'], label='Retorno Lag 2', linestyle='--', color='green')

plt.title('Retornos Logarítmicos del Oro y Desplazamientos')
plt.xlabel('Fecha')
plt.ylabel('Retorno')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Línea horizontal en 0
plt.grid(True)
plt.legend()
plt.show()


# # Desviacion estandar y Transformaciones Matematicas

# In[17]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Símbolo del contrato de futuros del oro
gold_symbol = 'GC=F'

# Rango de fechas
start_date = '2020-01-01'
end_date = '2024-08-23'

# Descarga de datos del oro
gold_data = yf.download(gold_symbol, start=start_date, end=end_date)

# Calcular los retornos logarítmicos del precio de cierre
gold_data['Return'] = np.log(gold_data['Close'] / gold_data['Close'].shift(1))

# Eliminar los valores nulos (resultado del shift)
gold_data.dropna(inplace=True)

# Calcular la desviación estándar de los retornos
std_dev = gold_data['Return'].std()
print(f"Desviación estándar de los retornos: {std_dev}")

# Aplicar transformaciones matemáticas
gold_data['Log_Return'] = np.log(1 + gold_data['Return'])  # Transformación logarítmica
gold_data['Sqrt_Return'] = np.sqrt(gold_data['Return'].clip(lower=0))  # Raíz cuadrada (solo para valores no negativos)

# Mostrar los primeros 5 registros
print(gold_data[['Return', 'Log_Return', 'Sqrt_Return']].head())

# Graficar los retornos originales y las transformaciones
plt.figure(figsize=(14, 8))

# Gráfico de retornos originales
plt.subplot(3, 1, 1)
plt.plot(gold_data.index, gold_data['Return'], label='Retornos Originales', color='blue')
plt.title('Retornos Logarítmicos del Oro')
plt.ylabel('Retorno')
plt.grid(True)

# Gráfico de retornos transformados (log)
plt.subplot(3, 1, 2)
plt.plot(gold_data.index, gold_data['Log_Return'], label='Retornos Logarítmicos Transformados', color='orange')
plt.title('Transformación Logarítmica de Retornos')
plt.ylabel('Log(Return)')
plt.grid(True)

# Gráfico de retornos transformados (raíz cuadrada)
plt.subplot(3, 1, 3)
plt.plot(gold_data.index, gold_data['Sqrt_Return'], label='Raíz Cuadrada de Retornos', color='green')
plt.title('Transformación de Raíz Cuadrada de Retornos')
plt.ylabel('Raíz Cuadrada de Retorno')
plt.grid(True)

plt.tight_layout()
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Precio del oro del 2020-2024

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
print(gold_data.head())

# Grafico
plt.figure(figsize=(12,6))
plt.plot(gold_data['Close'], label='Precio de cierre')
plt.title('Precio del Oro en 2022')
plt.xlabel('Fecha')
plt.ylabel('Precio en USD')
plt.legend()
plt.show()


# # Inflacion en México del 94 hasta el dia de hoy

# In[14]:


import pandas as pd

ruta_archivo = r'C:\Users\User\Downloads\INFLACION.xlsx'
nombre_hoja = 'Sheet1'  # Nombre de la hoja que deseas importar

# Leer la hoja de Excel a un DataFrame
datos_excel = pd.read_excel(ruta_archivo, sheet_name=nombre_hoja)

# Mostrar las primeras filas del DataFrame para verificar que la importación fue exitosa
print(datos_excel)


# In[ ]:





import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import calendar

##########################
año_seleccionado = st.selectbox("Selecciona el año:", [2023, 2024])

if año_seleccionado == 2023:
    df_full = pd.read_csv("eddypro_2023_full_output_2025-02-24T092218_exp.csv", skiprows=[0, 2], index_col=0)
    df_biomet = pd.read_csv("eddypro_2023_biomet_2025-02-24T092218_exp.csv", skiprows=[1], index_col=0)
else:
    df_full = pd.read_csv("eddypro_2024_full_output_2025-02-24T162122_exp.csv", skiprows=[0, 2], index_col=0)
    df_biomet = pd.read_csv("eddypro_2024_biomet_2025-02-24T162122_exp.csv", skiprows=[1], index_col=0)

st.title(f"Resumen estadistico EddyPro - Año {año_seleccionado}")

# Eliminar filas donde el índice contiene "not_enough_data" en full output
df_full = df_full[~df_full.index.astype(str).str.contains("not_enough_data")]

# Reemplazar todos los -9999 por NaN en full output
df_full.replace(-9999, np.nan, inplace=True)

# DF de full output sin outliers
def reemplazar_outliers_por_nan(df):
    df_modificado = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Reemplazar valores atípicos por NaN
        df_modificado[col] = df[col].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

    return df_modificado

df_sin_outliers = reemplazar_outliers_por_nan(df_full)
####################################

#ARchivo Biomet
# Reemplazar todos los -9999 por NaN en biomet
df_biomet.replace(-9999, np.nan, inplace=True)
####DF biomet sin outliers
def reemplazar_outliers_por_nan(df):
    df_modificado = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_modificado[col] = df[col].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

    return df_modificado
df_biomet_sin_outliers = reemplazar_outliers_por_nan(df_biomet)

#Asegurar que ambos df tengan la columna datetime 
# Para df_sin_outliers
df_sin_outliers = df_sin_outliers.copy()
df_sin_outliers['datetime'] = pd.to_datetime(
    df_sin_outliers['date'].astype(str) + ' ' + df_sin_outliers['time'].astype(str)
)

# Para df_biomet_sin_outliers 
df_biomet_sin_outliers = df_biomet_sin_outliers.copy()
if 'date' not in df_biomet_sin_outliers.columns:
    df_biomet_sin_outliers.reset_index(inplace=True)

df_biomet_sin_outliers['datetime'] = pd.to_datetime(
    df_biomet_sin_outliers['date'].astype(str) + ' ' + df_biomet_sin_outliers['time'].astype(str)
)

# Hacer merge por datetime 
df_combinado = pd.merge(
    df_sin_outliers,
    df_biomet_sin_outliers.drop(columns=['date', 'time']),
    on='datetime',
    how='left'
)
df_combinado.set_index(df_combinado.columns[0], inplace=True)
df_combinado.drop(columns=['datetime'], inplace=True)
st.subheader(f"Data FullOutput y Biomet unidas sin outliers - Año {año_seleccionado}") 
st.dataframe(df_combinado)
############################
# Convertir date de index a columna normal
df_biomet = df_biomet.reset_index()

# dar mismo formato de date en ambos df (datetime)
df_full['date'] = pd.to_datetime(df_full['date'])
df_biomet['date'] = pd.to_datetime(df_biomet['date'])

# Seleccionar columnas de interés
cols_full = ['date', 'LE','co2_flux', 'co2_mean','h2o_flux','co2_strg','air_temperature','air_pressure','air_density','air_heat_capacity','ET','water_vapor_density','specific_humidity','RH','wind_speed','bowen_ratio']
cols_biomet = ['date', 'P_RAIN_1_1_1','RN_1_1_1', 'SWC_1_1_1', 'SWC_2_1_1', 'SWC_3_1_1', 'TS_1_1_1','TS_2_1_1','TS_3_1_1','SHF_1_1_1','SHF_2_1_1','SHF_3_1_1']

df_full_sel = df_full[cols_full]
df_biomet_sel = df_biomet[cols_biomet]

# Unir los DataFrames
df_merged = pd.merge(df_full_sel, df_biomet_sel, on='date', how='inner')
# Variables de temperatura en Kelvin que se deben convertir a Celsius
temp_vars = ['air_temperature', 'TS_1_1_1', 'TS_2_1_1', 'TS_3_1_1']
# Convertir de Kelvin a Celsius
for col in temp_vars:
    if col in df_merged.columns:
        df_merged[col] = df_merged[col] - 273.15
#convertir m a mm en precipitacion
df_merged['P_RAIN_1_1_1'] = df_merged['P_RAIN_1_1_1'] * 1000

# Asegurar que 'date' esté en datetime y sin hora
df_merged['date'] = pd.to_datetime(df_merged['date']).dt.normalize()

# Asegurar que la columna 'date' no tenga hora y sea tipo datetime
df_merged['date'] = pd.to_datetime(df_merged['date']).dt.date  # solo año-mes-día

# Crear listas de columnas
columnas = [col for col in df_merged.columns if col != 'date']
columnas_sin_lluvia = [col for col in columnas if col != 'P_RAIN_1_1_1']

# Diccionario de agregaciones
agg_dict = {}

# Agregar media, máximo y mínimo para todas las columnas excepto la lluvia
for col in columnas_sin_lluvia:
    agg_dict[f"{col}_mean"] = (col, 'mean')
    agg_dict[f"{col}_max"] = (col, 'max')
    agg_dict[f"{col}_min"] = (col, 'min')

# Agregar suma (acumulado) para la lluvia
agg_dict['P_RAIN_1_1_1_sum'] = ('P_RAIN_1_1_1', 'sum')

# Agrupar por fecha
df_resumen_diario = df_merged.groupby('date').agg(**agg_dict).reset_index()

# Asegurar formato datetime para el índice
df_resumen_diario['date'] = pd.to_datetime(df_resumen_diario['date'])
df_resumen_diario.set_index('date', inplace=True)

# Formato del índice: solo fecha (sin hora)
df_resumen_diario.index = df_resumen_diario.index.date
df_resumen_diario.index.name = 'date'

# Mostrar en Streamlit
st.subheader(f"Estadisticos descriptivos para variables de interes - Año {año_seleccionado}") 
st.dataframe(df_resumen_diario)

# Extraer columnas de promedio diario
df_means = df_resumen_diario.filter(regex='_mean$')
df_means['P_RAIN_1_1_1_sum'] = df_resumen_diario['P_RAIN_1_1_1_sum']

# Calcular error estándar TOTAL 
variables_numericas = df_merged.select_dtypes(include='number')

# std y n total
std_total = variables_numericas.std()
n_total = variables_numericas.count()

# Error estándar total
error_estandar_total = std_total / n_total.pow(0.5)
error_estandar_total = error_estandar_total.rename("Error estándar total")

# Pasar a DataFrame para visualización
df_error_total = error_estandar_total.reset_index()
df_error_total.columns = ['Variable', 'Error estándar total']

#Detectar valores atípicos y dejar el valor si es outlier, sino NaN
def detectar_outliers_con_valor(df):
    outliers = pd.DataFrame(index=df.index)
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers[col] = df[col].where((df[col] < lower) | (df[col] > upper))
    return outliers

df_outliers = detectar_outliers_con_valor(df_means)

# Mostrar resultados en Streamlit
st.subheader(f"Error estándar de cada variable - Año {año_seleccionado}")
st.dataframe(df_error_total)

st.subheader(f"Valores atípicos detectados (IQR 1.5) - Año {año_seleccionado}")
st.dataframe(df_outliers)
# Crear copia limpia sin outliers 
df_means_sin_outliers = df_means.copy()

# Reemplazar los outliers por NaN
for col in df_outliers.columns:
    outlier_mask = df_outliers[col].notna()
    df_means_sin_outliers.loc[outlier_mask, col] = pd.NA
st.subheader(f'Data Depurada - Año {año_seleccionado}')
st.dataframe(df_means_sin_outliers)

# Diccionario de nombres y unidades de las variables
var_info = {
    "LE_mean": "Balanza de energía latente (W/m²)",
    "co2_flux_mean": "Flujo de CO₂ (µmol/s·m²)",
    "co2_mean": "Concentración de CO₂ (µmol/mol)",
    "h2o_flux_mean": "Flujo de H₂O (mmol/s·m²)",
    "co2_strg_mean": "Almacenamiento de CO₂ (µmol/s·m²)",
    "air_temperature_mean": "Temperatura del aire (°C)",
    "air_pressure_mean": "Presión atmosférica (Pa)",
    "air_density_mean": "Densidad del aire (kg/m³)",
    "air_heat_capacity_mean": "Capacidad calorífica del aire (J/kg·K)",
    "ET_mean": "Evapotranspiración (mm/h)",
    "water_vapor_density_mean": "Densidad de vapor de agua (kg/m³)",
    "specific_humidity_mean": "Humedad específica (kg/kg)",
    "RH_mean": "Humedad relativa (%)",
    "wind_speed_mean": "Velocidad del viento (m/s)",
    "bowen_ratio_mean": "Razón de Bowen",
    "P_RAIN_1_1_1_sum": "Precipitación acumulada (mm)",
    "RN_1_1_1_mean": "Radiación neta (W/m²)",
    "SWC_1_1_1_mean": "Humedad del suelo a 5 cm (m³/m³)",
    "SWC_2_1_1_mean": "Humedad del suelo a 10 cm (m³/m³)",
    "SWC_3_1_1_mean": "Humedad del suelo a 20 cm (m³/m³)",
    "TS_1_1_1_mean": "Temperatura del suelo a 5 cm (°C)",
    "TS_2_1_1_mean": "Temperatura del suelo a 10 cm (°C)",
    "TS_3_1_1_mean": "Temperatura del suelo a 20 cm (°C)",
    "SHF_1_1_1_mean": "Flujo de calor del suelo a 5 cm (W/m²)",
    "SHF_2_1_1_mean": "Flujo de calor del suelo a 10 cm (W/m²)",
    "SHF_3_1_1_mean": "Flujo de calor del suelo a 20 cm (W/m²)",
}

st.subheader(f"Visualización de variables sin valores atípicos Año {año_seleccionado}")

# Selección de variable
target_var = st.selectbox("Seleccione la variable a visualizar:", df_means_sin_outliers.columns)

# Crear gráfico
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_means_sin_outliers.index,
    y=df_means_sin_outliers[target_var],
    mode='lines+markers',
    name=target_var
))

# Personalización del gráfico
fig.update_layout(
    title=f"{var_info.get(target_var, target_var)} Año {año_seleccionado}",
    xaxis_title="Fecha",
    yaxis_title=var_info.get(target_var, target_var),
    xaxis=dict(tickformat='%Y-%m-%d'),
    legend=dict(x=1, y=1),
    hovermode="x unified"
)

# Mostrar gráfico en Streamlit
st.plotly_chart(fig)
###############################################
#Creas grafico mensual horario de CO2
# Título de la app
st.subheader(f"Análisis de Flujo de CO₂ - Año {año_seleccionado}")

# 1. Eliminar valores atípicos
Q1 = df_full['co2_flux'].quantile(0.25)
Q3 = df_full['co2_flux'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean = df_full[(df_full['co2_flux'] >= lower_bound) & 
                 (df_full['co2_flux'] <= upper_bound)].copy()

# 2. Crear columna datetime (versión robusta)
try:
    df_clean['datetime'] = pd.to_datetime(df_clean['date'].astype(str) + ' ' + df_clean['time'].astype(str))
except KeyError:
    df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])

# 3. Extraer mes y hora
df_clean['mes'] = df_clean['datetime'].dt.strftime('%B')
df_clean['hora'] = df_clean['datetime'].dt.strftime('%H:%M')
df_clean['hora_dt'] = pd.to_datetime(df_clean['hora'], format='%H:%M')

# 4. Agrupar datos
# Calculamos tanto media como desviación estándar
df_grouped = df_clean.groupby(['mes', 'hora_dt'])['co2_flux'].agg(['mean', 'std']).reset_index()
df_grouped.rename(columns={'mean': 'co2_flux_mean', 'std': 'co2_flux_std'}, inplace=True)

# 5. Widget para selección de mes
meses_ordenados = sorted(df_grouped['mes'].unique(), 
                      key=lambda x: pd.to_datetime(x, format='%B').month)
mes_seleccionado = st.selectbox("Seleccione el mesprueba:", meses_ordenados)

# 6. Filtrar por mes
mes_df = df_grouped[df_grouped['mes'] == mes_seleccionado]

# 7. Crear gráfico con banda de confianza
fig = go.Figure()

# Línea principal
fig.add_trace(go.Scatter(
    x=mes_df['hora_dt'],
    y=mes_df['co2_flux_mean'],
    mode='lines+markers',
    name=mes_seleccionado,
    line=dict(color='blue')
))

# Sombreado ±1 desviación estándar
# Línea principal (azul profesional)
fig.add_trace(go.Scatter(
    x=mes_df['hora_dt'],
    y=mes_df['co2_flux_mean'],
    mode='lines+markers',
    name=mes_seleccionado,
    line=dict(color='rgb(33, 113, 181)', width=2),
    marker=dict(size=5)
))

# Banda de confianza suavizada (±1 desviación estándar)
fig.add_trace(go.Scatter(
    x=pd.concat([mes_df['hora_dt'], mes_df['hora_dt'][::-1]]),
    y=pd.concat([
        mes_df['co2_flux_mean'] + mes_df['co2_flux_std'],
        (mes_df['co2_flux_mean'] - mes_df['co2_flux_std'])[::-1]
    ]),
    fill='toself',
    fillcolor='rgba(33, 113, 181, 0.1)',  # azul suave y transparente
    line=dict(color='rgba(0,0,0,0)'),  # sin borde
    hoverinfo="skip",
    showlegend=False,
    name='Intervalo de Confianza (±1σ)'
))


# Configuración del gráfico
fig.update_layout(
    title=f"Flujo de CO₂ - {mes_seleccionado}",
    xaxis_title="Hora del día",
    yaxis_title="Flujo de CO₂ (µmol/m²/s)",
    xaxis=dict(
        tickformat='%H:%M',
        tickmode='linear',
        dtick=3600000 * 3  # cada 3 horas
    ),
    showlegend=False 
)

st.plotly_chart(fig, use_container_width=True)
##############

# Título de la aplicación
st.subheader(f'Análisis Mensual de Precipitación Acumulada - Año {año_seleccionado}')

# 1. Preparar los datos
df = df_resumen_diario.copy()
df.index = pd.to_datetime(df.index)  # Convertir el índice a datetime

# Extraer componentes de fecha
df['mes'] = df.index.month
df['mes_nombre'] = df.index.month_name()

# 2. Crear el boxplot mensual
fig = px.box(df, 
             x='mes_nombre', 
             y='P_RAIN_1_1_1_sum',
             title=f'Distribución Mensual de Precipitación Acumulada Diaria - Año {año_seleccionado}',
             labels={'P_RAIN_1_1_1_sum': 'Precipitación (mm)', 'mes_nombre': 'Mes'},
             color='mes_nombre')

# Ordenar los meses cronológicamente
meses_orden = list(calendar.month_name)[1:]
fig.update_layout(
    xaxis={'categoryorder': 'array', 'categoryarray': meses_orden},
    showlegend=False,
    xaxis_title='Mes',
    yaxis_title='Precipitación Acumulada Diaria (mm)',
    hovermode='x unified'
)

# 3. Mostrar el gráfico
st.plotly_chart(fig, use_container_width=True)

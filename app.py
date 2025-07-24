# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 10:28:43 2025

@author: jahop
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -------------------------------------------------
# CONFIGURACIÓN DE LA APP
# -------------------------------------------------
st.set_page_config(page_title="Portafolio - Javier Pérez", page_icon="📊", layout="wide")

# -------------------------------------------------
# SIDEBAR (Contacto)
# -------------------------------------------------
st.sidebar.title("Contacto")
st.sidebar.markdown("""
**Email:** tu_correo@gmail.com  
**LinkedIn:** [linkedin.com/in/tuusuario](https://linkedin.com/in/tuusuario)  
**Teléfono:** +52 55 7425 5593
""")

# -------------------------------------------------
# TÍTULO PRINCIPAL
# -------------------------------------------------
st.title("Portafolio Profesional")
st.markdown("""
### Javier Horacio Pérez Ricárdez
**Matemático con Maestría en Ciencias de la Computación**  
Especialista en análisis de datos, visualización y desarrollo de soluciones tecnológicas aplicadas a **finanzas públicas y optimización de presupuestos**.
""")

# -------------------------------------------------
# SECCIÓN: PROYECTO DESTACADO (Datos Simulados)
# -------------------------------------------------
st.header("Proyecto Destacado: Análisis de Presupuesto Simulado")

# Simular datos de presupuesto por unidad responsable
unidades = [
    "Educación", "Salud", "Seguridad", "Transporte",
    "Cultura", "Obras Públicas", "Medio Ambiente",
    "Desarrollo Económico", "Turismo", "Vivienda"
]
presupuesto = [1800, 2200, 3500, 1500, 800, 2000, 1200, 1100, 900, 1000]  # en millones
ejercido = [1600, 2000, 2800, 1300, 600, 1700, 900, 950, 700, 850]        # en millones

# Crear DataFrame
df = pd.DataFrame({
    "UnidadResponsable": unidades,
    "Presupuesto": presupuesto,
    "Ejercido": ejercido
})
df["% Ejecutado"] = (df["Ejercido"] / df["Presupuesto"]) * 100

# Mostrar tabla
st.subheader("Tabla de Presupuesto Simulado")
st.dataframe(df)

# -------------------------------------------------
# VISUALIZACIÓN 1: Top 10 Presupuesto
# -------------------------------------------------
st.subheader("Top 10 Unidades por Presupuesto Asignado")
fig_top10 = px.bar(
    df.sort_values("Presupuesto", ascending=False),
    x="Presupuesto",
    y="UnidadResponsable",
    orientation="h",
    title="Top 10 Unidades por Presupuesto",
    color="Presupuesto",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig_top10, use_container_width=True)

# -------------------------------------------------
# VISUALIZACIÓN 2: Comparativa Asignado vs Ejercido
# -------------------------------------------------
st.subheader("Comparativa Presupuesto Asignado vs Ejecutado")
fig_cmp = px.scatter(
    df,
    x="Presupuesto",
    y="Ejercido",
    size="Presupuesto",
    color="% Ejecutado",
    hover_data=["UnidadResponsable", "% Ejecutado"],
    title="Asignado vs Ejecutado (Simulado)",
    color_continuous_scale="RdYlGn"
)
st.plotly_chart(fig_cmp, use_container_width=True)

# -------------------------------------------------
# NUEVA SECCIÓN: MODELO DE MACHINE LEARNING
# -------------------------------------------------
st.header("Modelo de Machine Learning: Predicción de Presupuesto Ejecutado")

# Preparar datos para el modelo
X = np.array(df["Presupuesto"]).reshape(-1, 1)  # Feature: Presupuesto
y = np.array(df["Ejercido"])  # Target: Ejecutado

# Entrenar modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Predicciones
y_pred = modelo.predict(X)

# Calcular métricas
mae = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
r2 = r2_score(y, y_pred)

# Mostrar métricas en DataFrame
metricas_df = pd.DataFrame({
    "Métrica": ["MAE", "RMSE", "R²"],
    "Valor": [mae, rmse, r2]
})

st.subheader("Métricas del Modelo")
st.dataframe(metricas_df)

# Gráfica del modelo
st.subheader("Regresión Lineal: Presupuesto vs Ejecutado")
fig_modelo = px.scatter(
    df,
    x="Presupuesto",
    y="Ejercido",
    title="Modelo Lineal - Predicción",
    trendline="ols",
    color="UnidadResponsable"
)
st.plotly_chart(fig_modelo, use_container_width=True)

# -------------------------------------------------
# MENSAJE FINAL
# -------------------------------------------------
st.markdown("---")
st.markdown("""
**¿Te interesa colaborar o conocer más de mis proyectos?**  
Puedes contactarme directamente por correo o LinkedIn.
""")

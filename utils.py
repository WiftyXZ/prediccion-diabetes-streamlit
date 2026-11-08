"""Funciones de apoyo de la aplicación: variables del modelo, validación de entradas y predicción.

Están separadas de app.py para poder probarlas sin abrir la interfaz.
"""
import math

import numpy as np
import pandas as pd

# Orden exacto de las columnas con que se entrenó el pipeline (el mismo del notebook)
COLUMNAS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

ETIQUETAS_CLASE = {0: "Sin diabetes", 1: "Con diabetes"}

# Rangos permitidos: cubren todos los valores reales del dataset con algo de margen.
# "opcional" = en el entrenamiento esa variable tenía ceros imposibles que se trataron como datos no medidos,
# así que la app permite dejarla sin dato y el pipeline la reemplaza por la mediana del entrenamiento.
VARIABLES = {
    "Pregnancies": {
        "etiqueta": "Número de embarazos", "unidad": "", "minimo": 0, "maximo": 20,
        "entero": True, "opcional": False, "defecto": 2, "paso": 1, "formato": "%d",
        "ayuda": "Número entero de embarazos. Rango permitido: 0 a 20.",
    },
    "Glucose": {
        "etiqueta": "Glucosa a las 2 h de una prueba de tolerancia oral", "unidad": "mg/dL", "minimo": 40, "maximo": 300,
        "entero": False, "opcional": True, "defecto": 120.0, "paso": 1.0, "formato": "%.1f",
        "ayuda": "Concentración de glucosa en plasma. Rango permitido: 40 a 300 mg/dL.",
    },
    "BloodPressure": {
        "etiqueta": "Presión arterial diastólica", "unidad": "mm Hg", "minimo": 20, "maximo": 150,
        "entero": False, "opcional": True, "defecto": 70.0, "paso": 1.0, "formato": "%.1f",
        "ayuda": "Presión diastólica. Rango permitido: 20 a 150 mm Hg.",
    },
    "SkinThickness": {
        "etiqueta": "Grosor del pliegue cutáneo del tríceps", "unidad": "mm", "minimo": 5, "maximo": 100,
        "entero": False, "opcional": True, "defecto": 25.0, "paso": 1.0, "formato": "%.1f",
        "ayuda": "Grosor del pliegue en el tríceps. Rango permitido: 5 a 100 mm.",
    },
    "Insulin": {
        "etiqueta": "Insulina sérica a las 2 h", "unidad": "µU/mL", "minimo": 10, "maximo": 900,
        "entero": False, "opcional": True, "defecto": 100.0, "paso": 1.0, "formato": "%.1f",
        "ayuda": "Insulina en suero. Rango permitido: 10 a 900 µU/mL. En el dataset casi la mitad de las pacientes no tiene este dato.",
    },
    "BMI": {
        "etiqueta": "Índice de masa corporal (IMC)", "unidad": "kg/m²", "minimo": 15, "maximo": 70,
        "entero": False, "opcional": True, "defecto": 28.0, "paso": 0.1, "formato": "%.1f",
        "ayuda": "Peso en kg dividido por la estatura en metros al cuadrado. Rango permitido: 15 a 70.",
    },
    "DiabetesPedigreeFunction": {
        "etiqueta": "Función de pedigrí de diabetes (antecedentes familiares)", "unidad": "", "minimo": 0.05, "maximo": 2.5,
        "entero": False, "opcional": False, "defecto": 0.4, "paso": 0.01, "formato": "%.3f",
        "ayuda": "Índice que resume los antecedentes familiares de diabetes. Rango permitido: 0.05 a 2.5.",
    },
    "Age": {
        "etiqueta": "Edad", "unidad": "años", "minimo": 21, "maximo": 90,
        "entero": True, "opcional": False, "defecto": 30, "paso": 1, "formato": "%d",
        "ayuda": "Edad en años. El modelo se entrenó con mujeres de 21 años o más. Rango permitido: 21 a 90.",
    },
}


def _es_faltante(valor):
    return valor is None or (isinstance(valor, float) and math.isnan(valor))


def validar_entrada(valores):
    """Revisa tipos, rangos y datos obligatorios. Devuelve una lista de mensajes de error (vacía si todo está bien)."""
    errores = []
    for col in COLUMNAS:
        info = VARIABLES[col]
        nombre = info["etiqueta"]
        valor = valores.get(col)

        if _es_faltante(valor):
            if not info["opcional"]:
                errores.append(f"{nombre}: es obligatorio.")
            continue

        if isinstance(valor, (bool, np.bool_)):
            errores.append(f"{nombre}: debe ser un número.")
            continue
        try:
            x = float(valor)
        except (TypeError, ValueError):
            errores.append(f"{nombre}: debe ser un número.")
            continue
        if not math.isfinite(x):
            errores.append(f"{nombre}: debe ser un número finito.")
            continue
        if info["entero"] and not x.is_integer():
            errores.append(f"{nombre}: debe ser un número entero.")
            continue
        if x < info["minimo"] or x > info["maximo"]:
            unidad = f" {info['unidad']}" if info["unidad"] else ""
            extra = " El modelo solo se entrenó con mujeres de 21 años o más." if col == "Age" else ""
            errores.append(f"{nombre}: {x:g} está fuera del rango permitido ({info['minimo']:g} a {info['maximo']:g}{unidad}).{extra}")
    return errores


def construir_registro(valores):
    """Arma el DataFrame de una fila con las columnas en el orden del entrenamiento (NaN si el dato no se midió)."""
    fila = {}
    for col in COLUMNAS:
        valor = valores.get(col)
        fila[col] = np.nan if _es_faltante(valor) else float(valor)
    return pd.DataFrame([fila], columns=COLUMNAS)


def predecir(pipeline, valores):
    """Valida, predice y devuelve clase, probabilidad de diabetes y los datos que se imputaron con la mediana."""
    errores = validar_entrada(valores)
    if errores:
        raise ValueError(" | ".join(errores))
    registro = construir_registro(valores)
    clase = int(pipeline.predict(registro)[0])
    probabilidad = float(pipeline.predict_proba(registro)[0, 1])

    medianas = pipeline.named_steps["imputar"].statistics_
    imputadas = {col: float(medianas[i]) for i, col in enumerate(COLUMNAS) if pd.isna(registro.loc[0, col])}
    return {"clase": clase, "probabilidad": probabilidad, "imputadas": imputadas}

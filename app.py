"""Aplicación Streamlit: predicción de diabetes tipo 2 (prototipo académico).

Proyecto de aula de Minería de Datos, Universidad de la Costa (CUC).
La app NO entrena el modelo: carga el pipeline completo guardado por el notebook.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import COLUMNAS, ETIQUETAS_CLASE, VARIABLES, predecir, validar_entrada

st.set_page_config(page_title="Predicción de diabetes (prototipo académico)", page_icon="🩺", layout="wide")

BASE = Path(__file__).parent
COLS_CON_CEROS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
COLORES = {"Sin diabetes": "#4C72B0", "Con diabetes": "#DD8452"}


# ---------------------------------------------------------------- carga de archivos
@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE / "models" / "pipeline_diabetes.joblib")


@st.cache_data
def cargar_metricas():
    with open(BASE / "models" / "metricas_finales.json", encoding="utf-8") as archivo:
        return json.load(archivo)


@st.cache_data
def cargar_datos():
    df = pd.read_csv(BASE / "data" / "diabetes.csv")
    df[COLS_CON_CEROS] = df[COLS_CON_CEROS].replace(0, np.nan)   # ceros imposibles = datos no medidos
    df["Grupo"] = df["Outcome"].map({0: "Sin diabetes", 1: "Con diabetes"})
    return df


@st.cache_data
def cargar_prueba():
    return pd.read_csv(BASE / "data" / "conjunto_prueba.csv")


modelo = cargar_modelo()
metricas = cargar_metricas()
datos = cargar_datos()

# ---------------------------------------------------------------- encabezado y barra lateral
st.title("🩺 Predicción de diabetes tipo 2")
st.warning(
    "**Prototipo académico.** Esta aplicación es un ejercicio de un curso de Minería de Datos. "
    "No es un dispositivo médico y **no sustituye la decisión de un profesional de la salud**."
)

with st.sidebar:
    st.header("Proyecto de aula")
    st.write("Minería de Datos · Universidad de la Costa (CUC)")
    st.write("**Integrantes**")
    st.write("- Esteban Lopez\n- Gonzalez Ramirez Harold")
    st.caption("Modelo: " + metricas["modelo"])

tab_proyecto, tab_eda, tab_pred, tab_eval = st.tabs(
    ["Proyecto", "Exploración de datos", "Predicción", "Evaluación del modelo"])

# ---------------------------------------------------------------- pestaña 1: proyecto
with tab_proyecto:
    st.header("De qué trata el proyecto")
    st.markdown(
        "**Objetivo.** Construir un modelo de clasificación que, a partir de ocho variables clínicas y demográficas, "
        "estime si una paciente tiene diabetes tipo 2, y llevarlo a una aplicación donde se pueda consultar.\n\n"
        "**Pregunta.** ¿Se puede predecir la diabetes con esas variables y qué modelo lo hace mejor, cuidando sobre todo "
        "de no dejar pasar a pacientes que sí tienen la enfermedad (falsos negativos)?"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Pacientes en el dataset", f"{len(datos)}")
    c2.metric("Variables predictoras", f"{len(COLUMNAS)}")
    c3.metric("Con diabetes", f"{datos['Outcome'].mean() * 100:.1f} %")

    st.subheader("Fuente de los datos")
    st.markdown(
        "Conjunto **Pima Indians Diabetes Database**, recolectado originalmente por el National Institute of Diabetes and "
        "Digestive and Kidney Diseases (Smith et al., 1988). Contiene registros de mujeres de al menos 21 años de herencia Pima.\n\n"
        "- Fuente original: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
        "- Archivo que usamos: una copia del mismo conjunto en "
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv "
        "(viene sin encabezados; los nombres de columna los agregamos nosotros siguiendo el orden documentado)."
    )

    st.subheader("Modelo final")
    st.markdown(
        "**Regresión logística con `class_weight=\"balanced\"`**, dentro de un `Pipeline` que imputa con la mediana y estandariza. "
        "La elegimos entre regresión logística, árbol de decisión y la versión balanceada comparando recall, F1 y AUC en validación cruzada, "
        "no solo la exactitud (detalle en la pestaña *Evaluación del modelo*)."
    )

# ---------------------------------------------------------------- pestaña 2: EDA
with tab_eda:
    st.header("Tres gráficas del análisis exploratorio")
    st.caption("Los ceros imposibles de glucosa, presión, pliegue cutáneo, insulina e IMC se trataron como datos no medidos.")

    col_a, col_b = st.columns(2)
    with col_a:
        conteo = datos["Grupo"].value_counts().reindex(["Sin diabetes", "Con diabetes"])
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.bar(conteo.index, conteo.values, color=[COLORES[k] for k in conteo.index])
        for i, v in enumerate(conteo.values):
            ax.text(i, v + 8, f"{v} ({100 * v / conteo.sum():.1f}%)", ha="center")
        ax.set(title="Distribución de la variable objetivo", ylabel="Pacientes")
        ax.set_ylim(0, conteo.max() * 1.15)
        st.pyplot(fig)
        plt.close(fig)
        st.write(f"Hay {int(conteo['Sin diabetes'])} pacientes sin diabetes y {int(conteo['Con diabetes'])} con diabetes. "
                 "Como las clases están desbalanceadas, la exactitud sola no basta para evaluar el modelo.")

    with col_b:
        cuartiles, bordes = pd.qcut(datos["Glucose"], 4, retbins=True)
        prevalencia = datos.groupby(cuartiles, observed=True)["Outcome"].mean() * 100
        etiquetas = [f"hasta {int(bordes[1])}"] + [f"{int(bordes[i]) + 1} a {int(bordes[i + 1])}" for i in range(1, 4)]
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.bar(range(4), prevalencia.values, color="#DD8452")
        ax.set_xticks(range(4))
        ax.set_xticklabels(etiquetas)
        for i, v in enumerate(prevalencia.values):
            ax.text(i, v + 1.5, f"{v:.1f}%", ha="center")
        ax.set(title="% con diabetes por cuartil de glucosa", xlabel="Glucosa (mg/dL)", ylabel="% con diabetes")
        ax.set_ylim(0, 85)
        st.pyplot(fig)
        plt.close(fig)
        st.write(f"El porcentaje de pacientes con diabetes pasa de {prevalencia.iloc[0]:.1f} % en el cuartil más bajo de glucosa "
                 f"a {prevalencia.iloc[-1]:.1f} % en el más alto: la glucosa es la variable más asociada con la diabetes.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, titulo in zip(axes, ["Glucose", "BMI"], ["Glucosa (mg/dL)", "IMC (kg/m²)"]):
        grupos = [datos.loc[datos["Grupo"] == g, col].dropna() for g in ["Sin diabetes", "Con diabetes"]]
        caja = ax.boxplot(grupos, patch_artist=True, tick_labels=["Sin diabetes", "Con diabetes"])
        for parche, g in zip(caja["boxes"], ["Sin diabetes", "Con diabetes"]):
            parche.set_facecolor(COLORES[g])
        ax.set(title=titulo)
    st.pyplot(fig)
    plt.close(fig)
    med = datos.groupby("Grupo")[["Glucose", "BMI"]].median()
    st.write(f"La mediana de glucosa es {med.loc['Con diabetes', 'Glucose']:.0f} en el grupo con diabetes frente a "
             f"{med.loc['Sin diabetes', 'Glucose']:.0f} en el otro; en IMC es {med.loc['Con diabetes', 'BMI']:.1f} frente a "
             f"{med.loc['Sin diabetes', 'BMI']:.1f}. Son asociaciones, no demuestran que una variable cause la diabetes.")

# ---------------------------------------------------------------- pestaña 3: predicción
with tab_pred:
    st.header("Consulta una predicción")
    st.caption("Ingresa los datos de una paciente. Si algún dato no se midió, marca «No tengo este dato»: "
               "el modelo lo reemplaza por la mediana de los datos de entrenamiento.")

    with st.form("formulario"):
        columnas_form = st.columns(2)
        entradas, faltantes = {}, {}
        for i, col in enumerate(COLUMNAS):
            info = VARIABLES[col]
            etiqueta = f"{info['etiqueta']} ({info['unidad']})" if info["unidad"] else info["etiqueta"]
            with columnas_form[i % 2]:
                if info["entero"]:
                    entradas[col] = st.number_input(etiqueta, value=int(info["defecto"]), step=1, key=f"in_{col}", help=info["ayuda"])
                else:
                    entradas[col] = st.number_input(etiqueta, value=float(info["defecto"]), step=float(info["paso"]),
                                                    format=info["formato"], key=f"in_{col}", help=info["ayuda"])
                if info["opcional"]:
                    faltantes[col] = st.checkbox("No tengo este dato", key=f"na_{col}")
        enviar = st.form_submit_button("Calcular predicción")

    if "ultimo_resultado" not in st.session_state:
        st.session_state["ultimo_resultado"] = None

    if enviar:
        valores = {col: (None if faltantes.get(col) else entradas[col]) for col in COLUMNAS}
        errores = validar_entrada(valores)
        st.session_state["ultimo_resultado"] = None
        if errores:
            st.error("**Revisa los datos ingresados** (no se calculó la predicción):\n\n" + "\n".join(f"- {e}" for e in errores))
        else:
            resultado = predecir(modelo, valores)
            st.session_state["ultimo_resultado"] = resultado
            etiqueta_clase = ETIQUETAS_CLASE[resultado["clase"]]
            r1, r2 = st.columns(2)
            r1.metric("Clasificación del modelo", etiqueta_clase)
            r2.metric("Probabilidad estimada de diabetes", f"{resultado['probabilidad'] * 100:.1f} %")
            st.progress(min(max(resultado["probabilidad"], 0.0), 1.0))
            st.caption("El modelo clasifica como «Con diabetes» cuando la probabilidad estimada supera 50 %.")
            if resultado["imputadas"]:
                detalle = ", ".join(f"{VARIABLES[c]['etiqueta'].lower()} (mediana de entrenamiento: {v:g})"
                                    for c, v in resultado["imputadas"].items())
                st.info(f"Datos no medidos reemplazados por la mediana: {detalle}. Con muchos datos faltantes la predicción es menos confiable.")
            st.caption("Resultado de un prototipo académico: no es un diagnóstico ni reemplaza una evaluación médica.")

    st.divider()
    with st.expander("Comprobar con un registro conocido del conjunto de prueba"):
        prueba = cargar_prueba()
        st.write("Elige un registro del conjunto de prueba. La app calcula su predicción con el pipeline guardado y la compara "
                 "con la que dio el notebook para ese mismo registro.")
        indice = st.number_input("Número de registro de prueba", min_value=0, max_value=len(prueba) - 1, value=0, step=1, key="idx_prueba")
        fila = prueba.iloc[int(indice)]
        valores_prueba = {col: (None if pd.isna(fila[col]) else float(fila[col])) for col in COLUMNAS}
        resultado_prueba = predecir(modelo, valores_prueba)
        clase_nb = int(fila["prediccion_notebook"])
        proba_nb = float(fila["probabilidad_notebook"])
        coincide = bool(resultado_prueba["clase"] == clase_nb and abs(resultado_prueba["probabilidad"] - proba_nb) < 1e-9)
        st.session_state["comparacion_prueba"] = {
            "indice": int(indice), "clase_app": resultado_prueba["clase"], "proba_app": resultado_prueba["probabilidad"],
            "clase_notebook": clase_nb, "proba_notebook": proba_nb, "coincide": coincide,
        }
        st.dataframe(pd.DataFrame([fila[COLUMNAS]]).rename(columns=lambda c: VARIABLES[c]["etiqueta"]), hide_index=True)
        comparacion = pd.DataFrame({
            "Clasificación": [ETIQUETAS_CLASE[resultado_prueba["clase"]], ETIQUETAS_CLASE[clase_nb]],
            "Probabilidad de diabetes": [f"{resultado_prueba['probabilidad']:.6f}", f"{proba_nb:.6f}"],
        }, index=["Esta aplicación", "Notebook"])
        st.table(comparacion)
        st.write(f"Diagnóstico real registrado en el dataset: **{ETIQUETAS_CLASE[int(fila['Outcome_real'])]}**.")
        if coincide:
            st.success("La predicción de la aplicación coincide con la del notebook para este registro.")
        else:
            st.error("La predicción de la aplicación NO coincide con la del notebook.")

# ---------------------------------------------------------------- pestaña 4: evaluación
with tab_eval:
    prueba_m = metricas["prueba"]
    vn, fp = metricas["matriz_confusion"][0]
    fn, vp = metricas["matriz_confusion"][1]
    cv = metricas["validacion_cruzada"]

    st.header("Evaluación del modelo final")
    st.write(f"Modelo: **{metricas['modelo']}**. Entrenado con {metricas['n_entrenamiento']} pacientes y evaluado con "
             f"{metricas['n_prueba']} que no vio durante el entrenamiento ({metricas['positivos_prueba']} con diabetes).")

    st.subheader("Métricas en el conjunto de prueba")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Exactitud", f"{prueba_m['exactitud']:.3f}")
    m2.metric("Precisión (con diabetes)", f"{prueba_m['precision_clase1']:.3f}")
    m3.metric("Recall (con diabetes)", f"{prueba_m['recall_clase1']:.3f}")
    m4.metric("F1 (con diabetes)", f"{prueba_m['f1_clase1']:.3f}")
    m5.metric("AUC", f"{prueba_m['auc']:.3f}")

    col_cm, col_tab = st.columns([1, 1])
    with col_cm:
        st.subheader("Matriz de confusión")
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        matriz = np.array(metricas["matriz_confusion"])
        ax.imshow(matriz, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matriz[i, j]), ha="center", va="center", fontsize=16,
                        color="white" if matriz[i, j] > matriz.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Sin diabetes", "Con diabetes"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Sin diabetes", "Con diabetes"])
        ax.set(xlabel="Predicción", ylabel="Real")
        st.pyplot(fig)
        plt.close(fig)
    with col_tab:
        st.subheader("Resultados por clase")
        por_clase = pd.DataFrame({
            "Precisión": [metricas["por_clase"][k]["precision"] for k in ["0", "1"]],
            "Recall": [metricas["por_clase"][k]["recall"] for k in ["0", "1"]],
            "F1": [metricas["por_clase"][k]["f1-score"] for k in ["0", "1"]],
            "Pacientes": [int(metricas["por_clase"][k]["support"]) for k in ["0", "1"]],
        }, index=["Sin diabetes", "Con diabetes"])
        st.dataframe(por_clase.style.format({"Precisión": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}))
        st.write(f"De las {fn + vp} pacientes con diabetes del conjunto de prueba, el modelo detectó {vp} y dejó pasar {fn} "
                 f"(falsos negativos). Además marcó como enfermas a {fp} de las {vn + fp} pacientes sin diabetes (falsos positivos).")

    st.subheader("Validación cruzada de 5 pliegues (media ± desviación)")
    tabla_cv = pd.DataFrame({
        "Métrica": ["Exactitud", "Precisión", "Recall", "F1", "AUC"],
        "Valor": [f"{cv[k]['media']:.3f} ± {cv[k]['desviacion']:.3f}" for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]],
    })
    st.dataframe(tabla_cv, hide_index=True)
    st.write("Recall en cada pliegue: " + ", ".join(f"{v:.3f}" for v in metricas["recall_por_pliegue"]) + ".")

    st.subheader("Por qué elegimos este modelo")
    comp = pd.DataFrame(metricas["comparacion_modelos_cv"]).T.rename(columns={
        "accuracy": "Exactitud", "precision": "Precisión", "recall": "Recall", "f1": "F1", "roc_auc": "AUC"})
    st.dataframe(comp.style.format("{:.3f}"))
    st.caption("Validación cruzada de 5 pliegues. Elegimos la versión balanceada por su recall y su F1, no por la exactitud.")

    st.subheader("Limitaciones")
    st.markdown(
        f"- **Deja pasar casos:** en la prueba no detectó {fn} de {fn + vp} pacientes con diabetes (recall {prueba_m['recall_clase1']:.3f}), "
        f"y en uno de los pliegues de la validación cruzada el recall fue de {min(metricas['recall_por_pliegue']):.3f}.\n"
        f"- **Falsos positivos:** marca como enferma a {fp} de {vn + fp} pacientes sanas del conjunto de prueba.\n"
        "- **Depende mucho de la glucosa**, que es parte de lo que se usa para diagnosticar diabetes; sirve poco para detectar la enfermedad antes de esa prueba.\n"
        "- **Datos pequeños y de una sola población:** 768 mujeres de al menos 21 años de herencia Pima. No se puede generalizar a otras poblaciones ni a hombres.\n"
        "- **Datos faltantes:** casi la mitad de las pacientes no tiene insulina y cerca de un 30 % no tiene pliegue cutáneo; esos datos se reemplazan por la mediana, "
        "y una predicción con varios datos faltantes es menos confiable.\n"
        "- **Sin validación externa** ni calibración de las probabilidades; la probabilidad mostrada es una estimación del modelo, no un riesgo clínico real.\n"
        "- **Asociación, no causalidad.** Es un prototipo académico y no sustituye una decisión profesional."
    )

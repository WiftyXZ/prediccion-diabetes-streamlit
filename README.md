# Predicción de diabetes tipo 2 (prototipo académico)

> **Prototipo académico.** Este proyecto es un ejercicio de un curso de Minería de Datos. No es un dispositivo médico y **no sustituye la decisión de un profesional de la salud**.

Proyecto de aula de Minería de Datos, Universidad de la Costa (CUC), tercer corte (Actividad 1: integración y despliegue con Streamlit).

## Integrantes
- Esteban Lopez
- Gonzalez Ramirez Harold

## Qué hace
Aplicación web en Streamlit que clasifica si una paciente tiene diabetes a partir de ocho variables clínicas y demográficas, y muestra la probabilidad estimada. Incluye la presentación del proyecto, tres gráficas del EDA, un formulario con validación de datos, las métricas finales, la matriz de confusión y las limitaciones del modelo.

## Datos
- **Pima Indians Diabetes Database** (Smith et al., 1988): 768 pacientes (mujeres de al menos 21 años de herencia Pima), 8 predictoras y la variable objetivo `Outcome` (1 = tiene diabetes).
- Fuente original: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Archivo que usamos (`data/diabetes.csv`): una copia del mismo conjunto descargada de https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv, que viene sin encabezados. Los nombres de columna los agregamos nosotros siguiendo el orden documentado. No la descargamos directamente de Kaggle.
- No hay datos personales: es un conjunto público y anónimo.

## Modelo final y métricas
**Regresión logística con `class_weight="balanced"`**, dentro de un `Pipeline` de scikit-learn (1.8.0): imputación con la mediana, estandarización y modelo. Los ceros imposibles de glucosa, presión, pliegue cutáneo, insulina e IMC se tratan como datos no medidos. Entrenado con 614 pacientes y evaluado con 154 (54 con diabetes), partición 80/20 estratificada con semilla 42.

| Métrica | Prueba | Validación cruzada (5 pliegues) |
|---|---|---|
| Exactitud | 0.734 | 0.747 ± 0.040 |
| Precisión (con diabetes) | 0.603 | 0.623 ± 0.063 |
| Recall (con diabetes) | 0.704 | 0.727 ± 0.047 |
| F1 (con diabetes) | 0.650 | 0.669 ± 0.040 |
| AUC | 0.813 | 0.836 ± 0.020 |

Matriz de confusión en prueba (filas = real, columnas = predicción): sin diabetes 75 / 25; con diabetes 16 / 38. El modelo dejó pasar 16 de las 54 pacientes con diabetes. Recall por pliegue: 0.741, 0.722, 0.778, 0.642, 0.755.

**Por qué este modelo y no otro** (validación cruzada de 5 pliegues, no solo exactitud):

| Modelo | Exactitud | Precisión | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Regresión logística | 0.772 | 0.726 | 0.571 | 0.636 | 0.837 |
| Árbol de decisión (profundidad 4) | 0.728 | 0.612 | 0.638 | 0.617 | 0.768 |
| Regresión logística balanceada | 0.747 | 0.623 | 0.727 | 0.669 | 0.836 |

Elegimos la versión balanceada por su recall y su F1: en un tamizaje es más grave dejar pasar a una paciente con diabetes que enviar a una sana a un examen adicional. Su AUC es igual al de la regresión logística normal, o sea que no discrimina mejor: mueve el punto de corte hacia detectar más casos, a cambio de más falsos positivos y menos exactitud. El árbol de decisión tuvo un recall muy inestable entre pliegues.

## Cómo funciona
1. `Proyecto_Aula_Corte3_Diabetes.ipynb` reconstruye el pipeline, lo revisa con la ficha de calidad y lo guarda con `joblib` en `models/pipeline_diabetes.joblib`. También genera `models/metricas_finales.json` y `data/conjunto_prueba.csv` (los 154 registros de prueba con la predicción del notebook).
2. `app.py` **carga** ese archivo; no entrena nada cada vez que se ejecuta.
3. Cuando alguien envía el formulario, `utils.validar_entrada` revisa tipos, rangos y datos obligatorios. Si hay errores, no predice y muestra qué corregir. Si todo está bien, `utils.predecir` arma una fila con las columnas en el orden del entrenamiento y llama a `predict` y `predict_proba` del pipeline.
4. Los datos marcados como «No tengo este dato» llegan como `NaN` y el imputador del pipeline los reemplaza por la mediana de entrenamiento (igual que en el notebook).

## Estructura del repositorio
```
app.py                               aplicación Streamlit
utils.py                             variables, validación de entradas y predicción
requirements.txt                     versiones con las que se probó
Proyecto_Aula_Corte3_Diabetes.ipynb  notebook (revisión, ficha de calidad, exportación)
models/pipeline_diabetes.joblib      pipeline completo serializado
models/metricas_finales.json         métricas que muestra la app
data/diabetes.csv                    dataset
data/conjunto_prueba.csv             registros de prueba con la predicción del notebook
tests/test_app.py                    pruebas automáticas
```

## Cómo ejecutarlo
```
pip install -r requirements.txt
streamlit run app.py
```
Se probó con Python 3.12. Para publicar en Streamlit Community Cloud: crear la app desde este repositorio, rama principal, archivo `app.py`, y elegir Python 3.12 en *Advanced settings*.

## Pruebas realizadas
Se ejecutan con `python -m pytest tests -v` (10 pruebas; las de interfaz usan `streamlit.testing.v1.AppTest`).

| Prueba pedida | Qué se hizo | Resultado |
|---|---|---|
| Caso normal | Embarazos 2, glucosa 120, presión 70, pliegue 25, insulina 100, IMC 28, pedigrí 0.4, edad 30 | Predice; la probabilidad es igual (diferencia menor a 1e-12) a la del pipeline original |
| Entrada inválida | Glucosa -5, edad 15 (menor de 21) e IMC 200 | En los tres casos muestra un error que nombra la variable y **no** predice |
| Registro conocido | Registros 0, 1, 25, 77 y 153 del conjunto de prueba, y los 154 por código | Clase y probabilidad coinciden con las del notebook (diferencia menor a 1e-9) |
| Dato no medido | Insulina marcada como «No tengo este dato» | Se imputa con la mediana y la probabilidad coincide con la del pipeline |
| Datos reales | Los 768 registros del dataset (ceros como no medidos) | Todos pasan la validación de rangos |

Ejemplo de registro conocido: registro de prueba 0 (embarazos 7, glucosa 159, presión 64, IMC 27.4, pedigrí 0.294, edad 40; pliegue e insulina sin dato). El notebook y la app dan «Con diabetes» con probabilidad 0.7332; el diagnóstico real en el dataset es «Sin diabetes» (es un falso positivo del modelo).

## Limitaciones
- Deja pasar casos: 16 de 54 pacientes con diabetes en la prueba, y un pliegue de la validación cruzada con recall de 0.642.
- Marca como enfermas a 25 de 100 pacientes sanas del conjunto de prueba.
- Depende mucho de la glucosa, que es parte de lo que se usa para diagnosticar diabetes.
- Datos pequeños y de una sola población (mujeres de al menos 21 años de herencia Pima): no se puede generalizar.
- Cerca de la mitad de las pacientes no tiene insulina y cerca de un 30 % no tiene pliegue cutáneo; esos datos se imputan con la mediana.
- Sin validación externa ni calibración de probabilidades. Asociación, no causalidad.

## Evaluación del uso de inteligencia artificial

**Herramienta utilizada.** Claude (Anthropic), modelo Claude Sonnet 5.5, en la interfaz de chat de claude.ai, usando su entorno de ejecución de Python.

**Propósito.** Revisar el modelo del segundo corte con la ficha de calidad, escribir el notebook de revisión y exportación, y construir `app.py`, `utils.py`, las pruebas y este README.

**Prompt principal.** «Tengo que hacer la siguiente actividad», con el PDF de instrucciones «Despliegue en Streamlit» adjunto en el mismo mensaje, dentro de la conversación donde ya se había trabajado el proyecto. Después: «puedes continuar porfavor».

**Cómo verificamos la respuesta.** Lo siguiente lo ejecutó la herramienta en su propio entorno:
- El notebook se ejecutó completo (21 celdas, sin errores) y las cifras escritas en sus textos se compararon con las salidas reales y con `metricas_finales.json`.
- El pipeline guardado, al recargarse, da las mismas predicciones que el original sobre los 154 registros de prueba.
- Las 10 pruebas automáticas pasan. Como pasaron a la primera, se hizo una prueba de mutación: se rompió a propósito una probabilidad del conjunto de prueba y un rango de validación, y 5 de las 10 pruebas fallaron, o sea que sí detectan errores.
- El servidor de Streamlit arranca, responde `ok` en su endpoint de salud y no registra errores. Las versiones de `requirements.txt` son las instaladas y probadas (`pip check` sin conflictos).
- Las cifras que muestra la app (exactitud, recall, matriz de confusión) son las mismas del notebook.

**Lo que la herramienta no pudo verificar:** no abrió la interfaz en un navegador real (solo pruebas automáticas y el arranque del servidor) y no puede publicar en GitHub ni en Streamlit Community Cloud, así que la instalación en la nube y la URL pública se verifican fuera de la herramienta. Lista de comprobación del grupo en la URL pública (marcar al completarla):
- [ ] La app abre y se ven las cuatro pestañas y el aviso de prototipo académico.
- [ ] Caso normal: calcula la predicción y la probabilidad.
- [ ] Entrada inválida (por ejemplo glucosa -5): muestra el error y no predice.
- [ ] Registro de prueba 0: aparece el mensaje de que coincide con el notebook.

**Problemas encontrados.**
1. La primera versión del notebook de esta actividad tuvo un error al ejecutarse (usaba `LinearRegression` sin importarla); se detectó al ejecutar y se corrigió.
2. El entorno de trabajo de la herramienta se reinició a mitad de la tarea y se perdió el trabajo intermedio porque no estaba guardado. Se reconstruyó todo y se comprobó que las métricas coincidían con las obtenidas antes.
3. Al reconstruir, la herramienta escribió una línea innecesaria y sin sentido para armar la URL de descarga del dataset (un `.replace` sobre el texto); se vio antes de ejecutar y se simplificó.
4. En la sesión del segundo corte la herramienta no entregó el notebook que había construido. Por eso el notebook de esta actividad reconstruye el modelo con la misma semilla y partición en lugar de continuar un archivo previo.
5. Las pruebas pasaron a la primera, lo cual podía significar que no comprobaban nada; de ahí la prueba de mutación descrita arriba.

**Ajustes realizados.** Se corrigió la importación, se eliminó la línea innecesaria, y los rangos de validación se ajustaron para que cubran los 768 registros reales (hay una prueba que lo comprueba). Se revisó la regularización `C` con validación cruzada solo sobre el entrenamiento; la mejora posible (F1 de 0.682 con `C = 0.1` frente a 0.672 con `C = 1`) es menor que la desviación entre pliegues, así que **no se cambió el modelo**.

**Decisión final.** Se conservó la regresión logística balanceada sin cambios, junto con la aplicación, el pipeline serializado y las pruebas, porque es el modelo elegido en el análisis con validación cruzada y porque la app reproduce exactamente las predicciones del notebook. Todos los integrantes deben poder explicar el código entregado: la sección «Cómo funciona» resume el flujo de datos.

## Referencia
Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Symposium on Computer Applications in Medical Care*, 261-265.

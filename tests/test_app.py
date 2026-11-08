"""Pruebas de la aplicación. Se ejecutan con:  python -m pytest tests -v   (o  python tests/test_app.py)

Cubren lo que pide la actividad: un caso normal, una entrada inválida y registros conocidos del conjunto de prueba,
comparando siempre con el pipeline original guardado por el notebook.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from utils import COLUMNAS, construir_registro, predecir, validar_entrada  # noqa: E402

PIPELINE = joblib.load(RAIZ / "models" / "pipeline_diabetes.joblib")
PRUEBA = pd.read_csv(RAIZ / "data" / "conjunto_prueba.csv")
CASO_NORMAL = {"Pregnancies": 2, "Glucose": 120.0, "BloodPressure": 70.0, "SkinThickness": 25.0,
               "Insulin": 100.0, "BMI": 28.0, "DiabetesPedigreeFunction": 0.4, "Age": 30}


def abrir_app():
    return AppTest.from_file(str(RAIZ / "app.py"), default_timeout=90).run()


def enviar_formulario(at):
    boton = next(b for b in at.button if b.label == "Calcular predicción")
    return boton.click().run()


def poner_caso(at, valores):
    for col, v in valores.items():
        at.number_input(key=f"in_{col}").set_value(v)
    return at


# ------------------------------------------------------------------ validación (sin interfaz)
def test_validacion_acepta_caso_normal():
    assert validar_entrada(CASO_NORMAL) == []


def test_validacion_rechaza_valores_invalidos():
    assert any("Glucosa" in e for e in validar_entrada({**CASO_NORMAL, "Glucose": -5.0}))
    assert any("Edad" in e for e in validar_entrada({**CASO_NORMAL, "Age": 15}))          # menor de 21
    assert any("entero" in e for e in validar_entrada({**CASO_NORMAL, "Pregnancies": 2.5}))
    assert any("número" in e for e in validar_entrada({**CASO_NORMAL, "BMI": "abc"}))
    assert any("finito" in e for e in validar_entrada({**CASO_NORMAL, "BMI": float("inf")}))
    assert any("obligatorio" in e for e in validar_entrada({**CASO_NORMAL, "Age": None}))
    assert any("IMC" in e for e in validar_entrada({**CASO_NORMAL, "BMI": 150.0}))


def test_variables_opcionales_pueden_faltar():
    sin_datos = {**CASO_NORMAL, "Insulin": None, "SkinThickness": float("nan")}
    assert validar_entrada(sin_datos) == []
    assert predecir(PIPELINE, sin_datos)["imputadas"].keys() == {"Insulin", "SkinThickness"}


def test_todo_el_dataset_pasa_la_validacion():
    datos = pd.read_csv(RAIZ / "data" / "diabetes.csv")
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        datos[col] = datos[col].replace(0, np.nan)
    for _, fila in datos.iterrows():
        valores = {c: (None if pd.isna(fila[c]) else float(fila[c])) for c in COLUMNAS}
        assert validar_entrada(valores) == [], valores


# ------------------------------------------------------------------ coincidencia con el notebook (sin interfaz)
def test_las_154_predicciones_coinciden_con_el_notebook():
    for _, fila in PRUEBA.iterrows():
        valores = {c: (None if pd.isna(fila[c]) else float(fila[c])) for c in COLUMNAS}
        r = predecir(PIPELINE, valores)
        assert r["clase"] == int(fila["prediccion_notebook"])
        assert abs(r["probabilidad"] - float(fila["probabilidad_notebook"])) < 1e-9


# ------------------------------------------------------------------ interfaz
def test_la_app_abre_sin_errores():
    at = abrir_app()
    assert not at.exception
    assert len(at.tabs) == 4
    assert any("Prototipo académico" in w.value for w in at.warning)


def test_app_caso_normal():
    at = poner_caso(abrir_app(), CASO_NORMAL)
    at = enviar_formulario(at)
    assert not at.exception
    resultado = at.session_state["ultimo_resultado"]
    assert resultado is not None
    esperado = float(PIPELINE.predict_proba(construir_registro(CASO_NORMAL))[0, 1])
    assert abs(resultado["probabilidad"] - esperado) < 1e-12
    assert resultado["clase"] == int(PIPELINE.predict(construir_registro(CASO_NORMAL))[0])
    assert len(at.error) == 0


def test_app_entrada_invalida_no_predice():
    for campo, valor, texto in [("Glucose", -5.0, "Glucosa"), ("Age", 15, "Edad"), ("BMI", 200.0, "IMC")]:
        at = poner_caso(abrir_app(), {**CASO_NORMAL, campo: valor})
        at = enviar_formulario(at)
        assert not at.exception
        assert at.session_state["ultimo_resultado"] is None
        assert len(at.error) == 1 and texto in at.error[0].value


def test_app_dato_no_medido_se_imputa():
    at = poner_caso(abrir_app(), CASO_NORMAL)
    at.checkbox(key="na_Insulin").set_value(True)
    at = enviar_formulario(at)
    resultado = at.session_state["ultimo_resultado"]
    assert resultado is not None and list(resultado["imputadas"]) == ["Insulin"]
    sin_insulina = {**CASO_NORMAL, "Insulin": None}
    assert abs(resultado["probabilidad"] - float(PIPELINE.predict_proba(construir_registro(sin_insulina))[0, 1])) < 1e-12


def test_app_registros_conocidos_coinciden_con_el_notebook():
    for indice in [0, 1, 25, 77, len(PRUEBA) - 1]:
        at = abrir_app()
        at.number_input(key="idx_prueba").set_value(indice).run()
        assert not at.exception
        comp = at.session_state["comparacion_prueba"]
        assert comp["indice"] == indice
        assert comp["coincide"] is True
        assert comp["clase_app"] == comp["clase_notebook"]
        assert abs(comp["proba_app"] - comp["proba_notebook"]) < 1e-9


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    fallos = 0
    for nombre, funcion in pruebas:
        try:
            funcion()
            print("OK    ", nombre)
        except Exception as error:   # noqa: BLE001
            fallos += 1
            print("FALLA ", nombre, "->", repr(error)[:300])
    print(f"\n{len(pruebas) - fallos} de {len(pruebas)} pruebas pasaron")
    sys.exit(1 if fallos else 0)

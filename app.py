# -*- coding: utf-8 -*-
# Bank Marketing - Análisis Exploratorio de Datos (EDA)
# Caso de Estudio N°1 | Especialización en Python for Analytics
# Para ejecutar:  streamlit run app.py

import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Bank Marketing - EDA", page_icon="🏦", layout="wide")

# ---------------------------------------------------------------
# DATOS DEL AUTOR  (cámbialos por los tuyos)
# ---------------------------------------------------------------
NOMBRE = "Tu Nombre Completo"
CURSO = "Especialización en Python for Analytics - DMC Institute"
ANIO = 2026

ORDEN_MESES = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


# ---------------------------------------------------------------
# MENÚ PRINCIPAL (sidebar)
# ---------------------------------------------------------------
st.sidebar.title("🏦 Bank Marketing")
menu = st.sidebar.selectbox("Menú", ["Home", "Carga del dataset", "EDA", "Conclusiones"])


# ===============================================================
# MÓDULO 1: HOME
# ===============================================================
if menu == "Home":
    st.title("Bank Marketing: Análisis Exploratorio de Datos")
    st.write(
        "Este proyecto analiza los datos de la última campaña de marketing de una "
        "institución financiera, cuya efectividad cayó de 12% a 8% en los últimos 6 meses. "
        "El objetivo es descubrir qué factores influyen en que un cliente acepte la campaña."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Nombre:** Harold Hernandez Medina")
        st.markdown("**Modulo:**  Python Fundamentals")
        st.markdown("**Edad:**  25 años")
        st.markdown("**Año:**  2026")
        st.markdown("**Descripcion:**  Una aplicación interactiva en Streamlit integrando los contenidos revisados en el módulo")
        st.markdown("**Tecnologias:**  GitHub,Streamlit,Python")
        st.write("Python, Pandas, NumPy, Matplotlib, Seaborn y Streamlit.")
    with col2:
        st.subheader("📊 Sobre el dataset")
        st.write(
            "El archivo BankMarketing.csv tiene una fila por cliente contactado, con su perfil "
            "(edad, trabajo, educación), su situación financiera, datos de la gestión comercial "
            "(canal, mes, duración, número de contactos), indicadores económicos y el resultado "
            "final en la columna **y** (yes = aceptó, no = no aceptó)."
        )


# ---------------------------------------------------------------
# FUNCIÓN PERSONALIZADA: separa variables numéricas y categóricas
# ---------------------------------------------------------------
def clasificar_variables(df):
    numericas = df.select_dtypes(include="number").columns.tolist()
    categoricas = df.select_dtypes(exclude="number").columns.tolist()
    return numericas, categoricas


# ---------------------------------------------------------------
# CLASE (POO): agrupa estadísticas, clasificación y gráficos
# ---------------------------------------------------------------
class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.numericas, self.categoricas = clasificar_variables(df)

    def estadisticas(self):
        """Describe + mediana + moda de las variables numéricas."""
        resumen = self.df[self.numericas].describe().T
        resumen["mediana"] = self.df[self.numericas].median()
        resumen["moda"] = self.df[self.numericas].mode().iloc[0]
        return resumen

    def histograma(self, col, bins=30, kde=True):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.df[col], bins=bins, kde=kde, ax=ax)
        ax.set_title(f"Distribución de {col}")
        return fig

    def barras(self, col, proporcion=False):
        conteo = self.df[col].value_counts(normalize=proporcion)
        if proporcion:
            conteo = conteo * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(conteo.index.astype(str), conteo.values)
        ax.set_title(f"{'Proporción (%)' if proporcion else 'Conteo'} de {col}")
        ax.tick_params(axis="x", rotation=45)
        return fig

    def tasa_aceptacion(self, col):
        """% de clientes que dijeron 'yes' en cada categoría de `col`."""
        return (self.df["y"] == "yes").groupby(self.df[col]).mean() * 100


# ===============================================================
# MÓDULO 2: CARGA DEL DATASET
# ===============================================================
elif menu == "Carga del dataset":
    st.title("📂 Carga del dataset")
    archivo = st.file_uploader("Sube el archivo BankMarketing.csv", type=["csv"])

    if archivo is not None:
        try:
            df = pd.read_csv(archivo, sep=";")  # el archivo usa ; como separador
        except Exception:
            df = None

        # Validamos que se haya leído bien (debe tener varias columnas y la columna 'y')
        if df is None or "y" not in df.columns or "age" not in df.columns:
            st.error("❌ El archivo no es válido. Verifica que sea BankMarketing.csv.")
        else:
            st.session_state["df"] = df  # lo guardamos para usarlo en otros módulos
            st.success("✅ Archivo cargado correctamente.")

    if "df" in st.session_state:
        df = st.session_state["df"]
        col1, col2 = st.columns(2)
        col1.metric("Filas", f"{df.shape[0]:,}")
        col2.metric("Columnas", df.shape[1])
        st.subheader("Vista previa (head)")
        st.dataframe(df.head())
    else:
        st.warning("Aún no has cargado ningún archivo.")


# ===============================================================
# MÓDULOS 3 y 4: solo funcionan si el dataset ya fue cargado
# ===============================================================
else:
    if "df" not in st.session_state:
        st.warning("⚠️ Primero carga el dataset en el módulo 'Carga del dataset'.")
        st.stop()
    df = st.session_state["df"]

    # -----------------------------------------------------------
    # MÓDULO 3: EDA
    # -----------------------------------------------------------
    if menu == "EDA":
        st.title("🔎 Análisis Exploratorio de Datos (EDA)")

        # Filtros en el sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("Filtros")
        edad_min, edad_max = int(df["age"].min()), int(df["age"].max())
        rango_edad = st.sidebar.slider("Rango de edad", edad_min, edad_max, (edad_min, edad_max))
        lista_canales = df["contact"].unique().tolist()
        canales = st.sidebar.multiselect("Canal de contacto", lista_canales, default=lista_canales)
        aplicar = st.sidebar.checkbox("Aplicar filtros", value=True)

        if aplicar:
            datos = df[(df["age"] >= rango_edad[0]) & (df["age"] <= rango_edad[1]) & (df["contact"].isin(canales))]
        else:
            datos = df

        if len(datos) == 0:
            st.error("Los filtros no devuelven datos. Cámbialos en el sidebar.")
            st.stop()

        an = DataAnalyzer(datos)  # objeto de la clase
        st.caption(f"Registros analizados: {len(datos):,} de {len(df):,}")

        tabs = st.tabs([
            "1. Info", "2. Variables", "3. Descriptivas", "4. Faltantes", "5. Numéricas",
            "6. Categóricas", "7. Num vs y", "8. Cat vs y", "9. Interactivo", "10. Hallazgos",
        ])

        # ---------- Ítem 1: Información general ----------
        with tabs[0]:
            st.subheader("Ítem 1: Información general del dataset")
            st.write("Estructura del dataset: `.info()`, tipos de datos y valores nulos.")
            col1, col2 = st.columns(2)
            with col1:
                buffer = io.StringIO()
                datos.info(buf=buffer)
                st.text(buffer.getvalue())
            with col2:
                tabla = pd.DataFrame({"Tipo": datos.dtypes.astype(str), "Nulos": datos.isnull().sum()})
                st.dataframe(tabla)

        # ---------- Ítem 2: Clasificación de variables ----------
        with tabs[1]:
            st.subheader("Ítem 2: Clasificación de variables")
            st.write("La función `clasificar_variables()` separa las columnas por tipo de dato.")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Variables numéricas", len(an.numericas))
                st.write(an.numericas)
            with col2:
                st.metric("Variables categóricas", len(an.categoricas))
                st.write(an.categoricas)

        # ---------- Ítem 3: Estadísticas descriptivas ----------
        with tabs[2]:
            st.subheader("Ítem 3: Estadísticas descriptivas")
            st.write("Resumen con `.describe()` más la mediana y la moda.")
            st.dataframe(an.estadisticas().round(2))

            var = st.selectbox("Elige una variable para interpretar", an.numericas, key="i3_var")
            media = datos[var].mean()
            mediana = datos[var].median()
            desv = datos[var].std()
            st.write(f"**{var}**: media = {media:.2f}, mediana = {mediana:.2f}, desviación estándar = {desv:.2f}.")
            if media > mediana * 1.05:
                st.write("La media es mayor que la mediana: hay valores altos que la 'jalan' hacia arriba (sesgo a la derecha).")
            elif media < mediana * 0.95:
                st.write("La media es menor que la mediana: hay valores bajos que la 'jalan' hacia abajo (sesgo a la izquierda).")
            else:
                st.write("La media y la mediana son parecidas: la distribución es aproximadamente simétrica.")

        # ---------- Ítem 4: Valores faltantes ----------
        with tabs[3]:
            st.subheader("Ítem 4: Análisis de valores faltantes")
            st.write("Además de los nulos, en este dataset los faltantes aparecen escritos como 'unknown'.")
            nulos = datos.isnull().sum()
            unknown = (datos == "unknown").sum()
            tabla = pd.DataFrame({"Nulos": nulos, "Unknown": unknown})
            tabla["% Unknown"] = (tabla["Unknown"] / len(datos) * 100).round(2)

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(tabla)
            with col2:
                con_unknown = tabla[tabla["Unknown"] > 0]["Unknown"]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(con_unknown.index, con_unknown.values)
                ax.set_title("Cantidad de 'unknown' por variable")
                st.pyplot(fig)

            peor = tabla["% Unknown"].idxmax()
            st.write(
                f"**Discusión:** hay {int(nulos.sum())} nulos reales. La variable con más 'unknown' es "
                f"**{peor}** ({tabla.loc[peor, '% Unknown']}%). Conviene mantener 'unknown' como una "
                f"categoría más, porque también puede aportar información."
            )

        # ---------- Ítem 5: Distribución de variables numéricas ----------
        with tabs[4]:
            st.subheader("Ítem 5: Distribución de variables numéricas")
            col1, col2, col3 = st.columns(3)
            var = col1.selectbox("Variable", an.numericas, key="i5_var")
            bins = col2.slider("Número de bins", 5, 100, 30, key="i5_bins")
            kde = col3.checkbox("Mostrar curva KDE", value=True, key="i5_kde")

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(an.histograma(var, bins, kde))
            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x=datos[var], ax=ax)
                ax.set_title(f"Boxplot de {var}")
                st.pyplot(fig)
            st.write(
                f"**Interpretación:** en {var} la media es {datos[var].mean():.2f} y la mediana "
                f"{datos[var].median():.2f}. Los puntos fuera de la caja del boxplot son valores atípicos."
            )

        # ---------- Ítem 6: Variables categóricas ----------
        with tabs[5]:
            st.subheader("Ítem 6: Análisis de variables categóricas")
            col1, col2 = st.columns(2)
            var = col1.selectbox("Variable categórica", an.categoricas, key="i6_var")
            prop = col2.checkbox("Mostrar proporciones (%)", key="i6_prop")

            conteo = datos[var].value_counts()
            tabla = pd.DataFrame({"Conteo": conteo, "Proporción (%)": (conteo / len(datos) * 100).round(2)})
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(tabla)
            with col2:
                st.pyplot(an.barras(var, prop))
            st.write(f"La categoría más frecuente de **{var}** es **{conteo.index[0]}** ({tabla.iloc[0, 1]}%).")

        # ---------- Ítem 7: Numérico vs categórico ----------
        with tabs[6]:
            st.subheader("Ítem 7: Análisis bivariado (numérico vs categórico)")
            st.write("Comparamos una variable numérica entre quienes aceptan (yes) y no aceptan (no).")
            var = st.selectbox("Variable numérica", an.numericas, key="i7_var")

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=datos, x="y", y=var, ax=ax)
                ax.set_title(f"{var} según y")
                st.pyplot(fig)
            with col2:
                resumen = datos.groupby("y")[var].agg(["mean", "median", "std"]).round(2)
                st.dataframe(resumen)

        # ---------- Ítem 8: Categórico vs categórico ----------
        with tabs[7]:
            st.subheader("Ítem 8: Análisis bivariado (categórico vs categórico)")
            st.write("¿En qué categorías es mayor el porcentaje de clientes que aceptan?")
            opciones = [c for c in an.categoricas if c != "y"]
            var = st.selectbox("Variable categórica", opciones, key="i8_var")

            tabla = pd.crosstab(datos[var], datos["y"], normalize="index") * 100
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                tabla.plot(kind="bar", stacked=True, ax=ax)
                ax.set_ylabel("% dentro de la categoría")
                ax.set_title(f"{var} vs y")
                st.pyplot(fig)
            with col2:
                st.dataframe(tabla.round(2))
            tasas = an.tasa_aceptacion(var)
            st.write(f"En **{var}**, la categoría con mayor aceptación es **{tasas.idxmax()}** ({tasas.max():.1f}%).")

        # ---------- Ítem 9: Análisis con parámetros ----------
        with tabs[8]:
            st.subheader("Ítem 9: Análisis basado en parámetros seleccionados")
            st.write("Elige las columnas y el estadístico para armar tu propio análisis.")

            col1, col2, col3 = st.columns(3)
            grupo = col1.selectbox("Agrupar por", an.categoricas, key="i9_grupo")
            metricas = col2.multiselect("Variables numéricas", an.numericas, default=["age", "duration"], key="i9_met")
            estadistico = col3.selectbox("Estadístico", ["mean", "median", "max", "min"], key="i9_est")

            if len(metricas) > 0:
                resultado = datos.groupby(grupo)[metricas].agg(estadistico)
                st.dataframe(resultado.round(2))
                st.bar_chart(resultado)
            else:
                st.info("Selecciona al menos una variable numérica.")

            st.markdown("**Matriz de correlación**")
            cols = st.multiselect("Variables a correlacionar", an.numericas,
                                  default=[c for c in an.numericas if c != "pdays"], key="i9_corr")
            if len(cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(datos[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)

        # ---------- Ítem 10: Hallazgos clave ----------
        with tabs[9]:
            st.subheader("Ítem 10: Hallazgos clave")
            tasa_global = (datos["y"] == "yes").mean() * 100

            tasa_contacto = an.tasa_aceptacion("contact")
            tasa_pout = an.tasa_aceptacion("poutcome")
            tasa_mes = an.tasa_aceptacion("month")
            tasa_mes = tasa_mes.reindex([m for m in ORDEN_MESES if m in tasa_mes.index])

            st.write(f"Efectividad global: **{tasa_global:.1f}%** de aceptación.")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**% de aceptación por canal**")
                st.bar_chart(tasa_contacto)
            with col2:
                st.write("**% de aceptación por resultado previo**")
                st.bar_chart(tasa_pout)
            with col3:
                st.write("**% de aceptación por mes**")
                st.bar_chart(tasa_mes)

            mediana_dur = datos.groupby("y")["duration"].median()
            st.markdown("**Insights principales**")
            st.write(f"- El canal con mejor aceptación es **{tasa_contacto.idxmax()}** ({tasa_contacto.max():.1f}%).")
            st.write(f"- Los clientes con éxito previo aceptan {tasa_pout.get('success', 0):.1f}%, frente a {tasa_pout.get('nonexistent', 0):.1f}% de los que no tenían historial.")
            st.write(f"- El mes con mejor aceptación es **{tasa_mes.idxmax()}** ({tasa_mes.max():.1f}%).")
            if "yes" in mediana_dur.index and "no" in mediana_dur.index:
                st.write(f"- La mediana de duración de llamada es {mediana_dur['yes']:.0f}s en quienes aceptan y {mediana_dur['no']:.0f}s en quienes no.")

    # -----------------------------------------------------------
    # MÓDULO 4: CONCLUSIONES (usa el dataset completo)
    # -----------------------------------------------------------
    else:
        st.title("✅ Conclusiones finales")
        an = DataAnalyzer(df)

        global_ = (df["y"] == "yes").mean() * 100
        t_contacto = an.tasa_aceptacion("contact")
        t_pout = an.tasa_aceptacion("poutcome")
        t_mes = an.tasa_aceptacion("month")
        una_vez = (df[df["campaign"] == 1]["y"] == "yes").mean() * 100
        muchas = (df[df["campaign"] > 10]["y"] == "yes").mean() * 100
        mediana_dur = df.groupby("y")["duration"].median()
        peso_mayo = (df["month"] == "may").mean() * 100
        peso_success = (df["poutcome"] == "success").mean() * 100

        st.markdown(f"### 1. El canal importa\n"
                    f"El celular logra {t_contacto.get('cellular', 0):.1f}% de aceptación y el teléfono fijo solo "
                    f"{t_contacto.get('telephone', 0):.1f}% (global: {global_:.1f}%). "
                    f"**Decisión:** priorizar el contacto por celular.")
        st.markdown(f"### 2. Los clientes con éxito previo son la mejor oportunidad\n"
                    f"Aceptan {t_pout.get('success', 0):.1f}% frente a {t_pout.get('nonexistent', 0):.1f}% de quienes "
                    f"no tenían historial, aunque son solo el {peso_success:.1f}% de la base. "
                    f"**Decisión:** priorizar su recontacto.")
        st.markdown(f"### 3. Insistir más no vende más\n"
                    f"Con 1 solo contacto la aceptación es {una_vez:.1f}%; con más de 10 contactos baja a {muchas:.1f}%. "
                    f"**Decisión:** poner un tope de intentos por cliente.")
        st.markdown(f"### 4. Las llamadas que terminan en venta son más largas\n"
                    f"La mediana de duración es {mediana_dur['yes']:.0f}s en quienes aceptan y {mediana_dur['no']:.0f}s en "
                    f"quienes no. La duración solo se conoce al terminar la llamada, así que sirve para evaluar el guion "
                    f"y capacitar a los ejecutivos, no para elegir a quién llamar.")
        st.markdown(f"### 5. La carga de llamadas está mal distribuida en el año\n"
                    f"Mayo concentra el {peso_mayo:.0f}% de los contactos y convierte solo {t_mes.get('may', 0):.1f}%, "
                    f"mientras que marzo convierte {t_mes.get('mar', 0):.1f}%. "
                    f"**Decisión:** redistribuir las llamadas durante el año.")

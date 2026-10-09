# -*- coding: utf-8 -*-
"""
Bank Marketing - Análisis Exploratorio de Datos (EDA) interactivo
Caso de Estudio N°1 | Especialización en Python for Analytics

Ejecutar con:  streamlit run app.py
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ======================================================================
# CONFIGURACIÓN GENERAL
# ======================================================================
st.set_page_config(
    page_title="Bank Marketing | EDA",
    page_icon="🏦",
    layout="wide",
)
sns.set_theme(style="whitegrid")

# >>> PERSONALIZA ESTOS DATOS <<<
AUTOR = {
    "nombre": "Tu Nombre Completo",
    "curso": "Especialización en Python for Analytics - DMC Institute",
    "anio": 2026,
}

ORDEN_MESES = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
ORDEN_DIAS = ["mon", "tue", "wed", "thu", "fri"]
ORDEN_EDUCACION = [
    "illiterate", "basic.4y", "basic.6y", "basic.9y",
    "high.school", "professional.course", "university.degree", "unknown",
]
COLORES_Y = {"no": "#B0B7C3", "yes": "#1F6FEB"}
AZUL = "#1F6FEB"
GRIS = "#B0B7C3"

COLUMNAS_ESPERADAS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "y",
]

MENU = [
    "🏠 Home",
    "📂 Carga del dataset",
    "🔎 Análisis Exploratorio (EDA)",
    "✅ Conclusiones finales",
]


# ======================================================================
# FUNCIONES AUXILIARES
# ======================================================================
def clasificar_variables(df):
    """Función personalizada: separa las columnas en numéricas y categóricas."""
    numericas = df.select_dtypes(include="number").columns.tolist()
    categoricas = [col for col in df.columns if col not in numericas]
    return {"numericas": numericas, "categoricas": categoricas}


def cargar_csv(archivo):
    """Lee el CSV detectando el separador (el dataset original usa ';')."""
    df = pd.read_csv(archivo, sep=None, engine="python")
    if df.shape[1] == 1:  # Respaldo por si no detectó el separador
        archivo.seek(0)
        df = pd.read_csv(archivo, sep=";")
    return df


def validar_dataset(df):
    """Verifica que el archivo tenga la estructura de BankMarketing."""
    if df.empty:
        return False, "El archivo está vacío."
    faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    if faltantes:
        return False, f"Faltan columnas esperadas: {', '.join(faltantes)}"
    if not set(df["y"].unique()) <= {"yes", "no"}:
        return False, "La columna 'y' debe contener únicamente 'yes' / 'no'."
    return True, "Dataset válido."


def mostrar(fig):
    """Muestra una figura de Matplotlib en Streamlit y libera memoria."""
    st.pyplot(fig)
    plt.close(fig)


def valor(tabla, clave, campo="tasa"):
    """Devuelve tabla.loc[clave, campo] o NaN si la categoría no existe."""
    try:
        return tabla.loc[clave, campo]
    except KeyError:
        return np.nan


# ======================================================================
# CLASE PRINCIPAL (POO)
# ======================================================================
class DataAnalyzer:
    """Encapsula estadísticas, clasificación de variables y visualizaciones."""

    def __init__(self, df, target="y"):
        self.df = df.copy()
        self.target = target
        tipos = clasificar_variables(self.df)
        self.numericas = tipos["numericas"]
        self.categoricas = tipos["categoricas"]
        self.cat_features = [c for c in self.categoricas if c != target]
        self.y_bin = (self.df[target] == "yes").astype(int)

    # ------------------------- Utilidades -------------------------
    def tasa_global(self):
        """Efectividad global: (ventas / base) x 100."""
        return self.y_bin.mean() * 100

    def _orden(self, col):
        presentes = set(self.df[col].unique())
        for orden in (ORDEN_MESES, ORDEN_DIAS, ORDEN_EDUCACION):
            if presentes <= set(orden):
                return [o for o in orden if o in presentes]
        return self.df[col].value_counts().index.tolist()

    def segmentar_edad(self):
        return pd.cut(
            self.df["age"], bins=[0, 25, 35, 45, 55, 65, 120],
            labels=["<=25", "26-35", "36-45", "46-55", "56-65", ">65"],
        )

    def segmentar_campaign(self):
        return pd.cut(
            self.df["campaign"], bins=[0, 1, 2, 3, 5, 10, 1000],
            labels=["1", "2", "3", "4-5", "6-10", ">10"],
        )

    # ------------------------- Información general -------------------------
    def info_texto(self):
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    def tabla_tipos(self):
        return pd.DataFrame({
            "Tipo de dato": self.df.dtypes.astype(str),
            "No nulos": self.df.notna().sum(),
            "Nulos": self.df.isna().sum(),
            "Valores únicos": self.df.nunique(),
        })

    def tabla_clasificacion(self):
        filas = [(c, "Numérica") for c in self.numericas]
        filas += [(c, "Categórica") for c in self.categoricas]
        return pd.DataFrame(filas, columns=["Variable", "Clasificación"])

    # ------------------------- Estadística descriptiva -------------------------
    def describe_numerico(self):
        return self.df[self.numericas].describe().T

    def resumen_central(self):
        s = self.df[self.numericas]
        tabla = pd.DataFrame({
            "Media": s.mean(),
            "Mediana": s.median(),
            "Moda": s.mode().iloc[0],
            "Desv. estándar": s.std(),
        })
        tabla["Coef. variación (%)"] = tabla["Desv. estándar"] / tabla["Media"].abs() * 100
        return tabla

    def interpretar_distribucion(self, col):
        s = self.df[col]
        media, mediana, std = s.mean(), s.median(), s.std()
        moda = s.mode().iloc[0]
        cv = std / abs(media) * 100 if media != 0 else np.nan

        if abs(media - mediana) <= 0.05 * abs(mediana):
            forma = "una distribución aproximadamente simétrica (media ≈ mediana)"
        elif media > mediana:
            forma = "un sesgo positivo, con cola a la derecha (media > mediana)"
        else:
            forma = "un sesgo negativo, con cola a la izquierda (media < mediana)"

        if np.isnan(cv):
            disp = "dispersión no calculable"
        elif cv < 15:
            disp = f"dispersión baja (CV = {cv:.1f}%)"
        elif cv < 50:
            disp = f"dispersión moderada (CV = {cv:.1f}%)"
        else:
            disp = f"dispersión alta (CV = {cv:.1f}%)"

        texto = (
            f"**{col}**: media = {media:,.2f}, mediana = {mediana:,.2f}, "
            f"moda = {moda:,.2f}. Presenta {forma} y {disp}."
        )
        if col == "pdays":
            texto += " ⚠️ El valor 999 significa *'nunca contactado antes'*, no son días reales."
        return texto

    # ------------------------- Faltantes -------------------------
    def tabla_faltantes(self):
        nulos = self.df.isna().sum()
        unknown = (self.df[self.categoricas] == "unknown").sum()
        unknown = unknown.reindex(self.df.columns, fill_value=0)
        tabla = pd.DataFrame({"Nulos (NaN)": nulos, "'unknown'": unknown})
        tabla["Total faltante"] = tabla["Nulos (NaN)"] + tabla["'unknown'"]
        tabla["% faltante"] = tabla["Total faltante"] / len(self.df) * 100
        return tabla

    def graf_faltantes(self, incluir_unknown=True):
        tabla = self.tabla_faltantes()
        col = "Total faltante" if incluir_unknown else "Nulos (NaN)"
        datos = tabla[tabla[col] > 0][col].sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        if datos.empty:
            ax.text(0.5, 0.5, "Sin valores faltantes", ha="center", va="center", fontsize=13)
            ax.axis("off")
        else:
            ax.barh(datos.index, datos.values, color=AZUL)
            for i, v in enumerate(datos.values):
                ax.text(v, i, f" {v:,}", va="center", fontsize=9)
            ax.set_xlabel("Cantidad de registros")
            ax.set_title("Valores faltantes por variable")
        fig.tight_layout()
        return fig

    # ------------------------- Tasas de conversión -------------------------
    def tasa_conversion(self, serie):
        """Tasa de aceptación (%) por cada grupo de `serie`."""
        t = self.y_bin.groupby(serie, observed=True).agg(["mean", "sum", "count"])
        t.columns = ["tasa", "aceptaron", "contactos"]
        t["tasa"] = t["tasa"] * 100
        return t

    def graf_tasa(self, tabla, titulo, ax=None, rotar=True):
        propio = ax is None
        if propio:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure
        etiquetas = tabla.index.astype(str)
        colores = [AZUL if t >= self.tasa_global() else GRIS for t in tabla["tasa"]]
        ax.bar(etiquetas, tabla["tasa"], color=colores)
        ax.axhline(self.tasa_global(), color="crimson", ls="--", lw=1.2,
                   label=f"Global: {self.tasa_global():.1f}%")
        for i, v in enumerate(tabla["tasa"]):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(titulo, fontsize=11)
        ax.set_ylabel("% de aceptación")
        ax.set_xlabel("")
        ax.legend(fontsize=8)
        if rotar:
            ax.tick_params(axis="x", rotation=45)
        if propio:
            fig.tight_layout()
        return fig

    # ------------------------- Univariado -------------------------
    def graf_histograma(self, col, bins=30, kde=True):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.df[col], bins=bins, kde=kde, color=AZUL, ax=ax)
        ax.axvline(self.df[col].mean(), color="crimson", ls="--", label="Media")
        ax.axvline(self.df[col].median(), color="green", ls="-.", label="Mediana")
        ax.set_title(f"Distribución de {col}")
        ax.legend()
        fig.tight_layout()
        return fig

    def graf_boxplot(self, col):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=self.df[col], color=AZUL, ax=ax)
        ax.set_title(f"Boxplot de {col}")
        fig.tight_layout()
        return fig

    def graf_conteo(self, col, proporcion=False):
        orden = self._orden(col)
        conteo = self.df[col].value_counts().reindex(orden)
        datos = conteo / conteo.sum() * 100 if proporcion else conteo
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(datos.index.astype(str), datos.values, color=AZUL)
        for i, v in enumerate(datos.values):
            ax.text(i, v, f"{v:.1f}%" if proporcion else f"{v:,}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{'Proporción (%)' if proporcion else 'Frecuencia'} de {col}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig

    # ------------------------- Bivariado -------------------------
    def graf_box_por_y(self, col, outliers=True):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=self.df, x=self.target, y=col, hue=self.target,
                    order=["no", "yes"], hue_order=["no", "yes"],
                    palette=COLORES_Y, showfliers=outliers, legend=False, ax=ax)
        ax.set_title(f"{col} según resultado (y)")
        fig.tight_layout()
        return fig

    def graf_hist_por_y(self, col, bins=30):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=self.df, x=col, hue=self.target, hue_order=["no", "yes"],
                     palette=COLORES_Y, bins=bins, stat="density",
                     common_norm=False, element="step", ax=ax)
        ax.set_title(f"Distribución de {col} por resultado (densidad)")
        fig.tight_layout()
        return fig

    def tabla_por_y(self, col):
        return self.df.groupby(self.target)[col].agg(["count", "mean", "median", "std", "min", "max"])

    def crosstab_y(self, col, normalizar=False):
        if normalizar:
            ct = pd.crosstab(self.df[col], self.df[self.target], normalize="index") * 100
        else:
            ct = pd.crosstab(self.df[col], self.df[self.target])
        ct = ct.reindex(columns=["no", "yes"], fill_value=0)
        return ct.reindex(self._orden(col))

    def graf_apiladas(self, col):
        ct = self.crosstab_y(col, normalizar=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ct.plot(kind="bar", stacked=True, ax=ax,
                color=[COLORES_Y["no"], COLORES_Y["yes"]])
        ax.set_ylabel("% dentro de la categoría")
        ax.set_xlabel("")
        ax.set_title(f"{col} vs y (100% apilado)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="y", loc="lower right")
        fig.tight_layout()
        return fig

    # ------------------------- Análisis dinámico -------------------------
    def graf_grupos(self, grupo, metricas, estadistico="mean", ordenar=False):
        tabla = self.df.groupby(grupo)[metricas].agg(estadistico)
        if ordenar:
            tabla = tabla.sort_values(metricas[0], ascending=False)
        fig, axes = plt.subplots(1, len(metricas), figsize=(5.5 * len(metricas), 4))
        axes = np.atleast_1d(axes)
        for ax, met in zip(axes, metricas):
            ax.bar(tabla.index.astype(str), tabla[met], color=AZUL)
            ax.set_title(f"{estadistico} de {met}")
            ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig, tabla

    def graf_dispersion(self, x, y, n=2000):
        muestra = self.df.sample(min(n, len(self.df)), random_state=42)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(data=muestra, x=x, y=y, hue=self.target,
                        hue_order=["no", "yes"], palette=COLORES_Y,
                        alpha=0.6, s=25, ax=ax)
        ax.set_title(f"{x} vs {y} (muestra de {len(muestra):,} registros)")
        fig.tight_layout()
        return fig

    def graf_correlacion(self, columnas):
        datos = self.df[columnas].copy()
        datos["y (1=yes)"] = self.y_bin
        corr = datos.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    linewidths=0.5, annot_kws={"size": 8}, ax=ax)
        ax.set_title("Matriz de correlación")
        fig.tight_layout()
        return fig

    # ------------------------- Resumen -------------------------
    def graf_resumen(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        paneles = [
            (self.df["contact"], "Canal de contacto", False),
            (self.df["poutcome"], "Resultado campaña anterior", False),
            (self.df["month"], "Mes de contacto", False),
            (self.segmentar_edad(), "Grupo de edad", False),
            (self.segmentar_campaign(), "N° de contactos en la campaña", False),
        ]
        for ax, (serie, titulo, rotar) in zip(axes.flat, paneles):
            tabla = self.tasa_conversion(serie)
            if serie.name == "month":
                tabla = tabla.reindex([m for m in ORDEN_MESES if m in tabla.index])
            self.graf_tasa(tabla, f"Aceptación por: {titulo}", ax=ax, rotar=rotar)
        cols = [c for c in self.numericas if c != "pdays"]
        datos = self.df[cols].copy()
        datos["y (1=yes)"] = self.y_bin
        sns.heatmap(datos.corr()[["y (1=yes)"]].drop("y (1=yes)").sort_values("y (1=yes)"),
                    annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=False, ax=axes[1, 2])
        axes[1, 2].set_title("Correlación con y")
        fig.suptitle("Resumen de hallazgos clave", fontsize=15, fontweight="bold")
        fig.tight_layout()
        return fig


# ======================================================================
# GENERACIÓN DE INSIGHTS (usa f-strings con los datos actuales)
# ======================================================================
def generar_insights(an):
    ins = []
    ins.append(
        f"La efectividad global en los datos analizados es **{an.tasa_global():.1f}%** "
        f"({int(an.y_bin.sum()):,} aceptaciones sobre {len(an.df):,} contactos); "
        f"el {100 - an.tasa_global():.1f}% restante respondió 'no'."
    )
    try:
        t = an.tasa_conversion(an.df["contact"])
        ins.append(
            f"**Canal:** '{t['tasa'].idxmax()}' convierte {t['tasa'].max():.1f}% frente a "
            f"{t['tasa'].min():.1f}% de '{t['tasa'].idxmin()}'."
        )
    except (KeyError, ValueError):
        pass
    try:
        t = an.tasa_conversion(an.df["poutcome"])
        peso = t.loc["success", "contactos"] / t["contactos"].sum() * 100
        ins.append(
            f"**Campaña anterior:** quienes ya compraron ('success') aceptan "
            f"{t.loc['success', 'tasa']:.1f}%, pero son solo {peso:.1f}% de la base."
        )
    except (KeyError, ValueError):
        pass
    try:
        med = an.df.groupby("y")["duration"].median()
        ins.append(
            f"**Duración:** la mediana de llamada es {med['yes']:.0f}s en quienes aceptan vs "
            f"{med['no']:.0f}s en quienes rechazan (indicador de interés, pero solo se conoce "
            f"al terminar la llamada)."
        )
    except (KeyError, ValueError):
        pass
    try:
        t = an.tasa_conversion(an.segmentar_campaign())
        ins.append(
            f"**Insistencia:** con 1 contacto la aceptación es {t['tasa'].iloc[0]:.1f}%; "
            f"con más de 10 contactos baja a {t['tasa'].iloc[-1]:.1f}%."
        )
    except (KeyError, ValueError, IndexError):
        pass
    try:
        t = an.tasa_conversion(an.df["month"])
        mes_vol = t["contactos"].idxmax()
        t100 = t[t["contactos"] >= 100]
        ins.append(
            f"**Estacionalidad:** el mes con más contactos es '{mes_vol}' "
            f"({t.loc[mes_vol, 'contactos'] / t['contactos'].sum() * 100:.0f}% del total) con una "
            f"aceptación de {t.loc[mes_vol, 'tasa']:.1f}%; el mejor mes (≥100 contactos) es "
            f"'{t100['tasa'].idxmax()}' con {t100['tasa'].max():.1f}%."
        )
    except (KeyError, ValueError):
        pass
    try:
        t = an.tasa_conversion(an.segmentar_edad())
        t100 = t[t["contactos"] >= 100]
        ins.append(
            f"**Edad:** el grupo con mayor aceptación es '{t100['tasa'].idxmax()}' "
            f"({t100['tasa'].max():.1f}%) y el de menor aceptación '{t100['tasa'].idxmin()}' ({t100['tasa'].min():.1f}%)."
        )
    except (KeyError, ValueError):
        pass
    return ins


# ======================================================================
# MÓDULO 1: HOME
# ======================================================================
def modulo_home():
    st.title("🏦 Bank Marketing: Análisis Exploratorio de Datos")
    st.markdown(
        "Herramienta interactiva para **entender qué factores influyen en la aceptación "
        "de las campañas de marketing** de una institución financiera, cuya efectividad "
        "cayó de **12% a 8%** en los últimos 6 meses."
    )
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Autor")
        st.markdown(
            f"- **Nombre:** {AUTOR['nombre']}\n"
            f"- **Curso:** {AUTOR['curso']}\n"
            f"- **Año:** {AUTOR['anio']}"
        )
        st.subheader("🎯 Objetivo del análisis")
        st.markdown(
            "Explorar relaciones y comportamientos relevantes entre las variables de la "
            "última campaña para apoyar **decisiones comerciales**. "
            "El proyecto **no** construye modelos predictivos."
        )
    with col2:
        st.subheader("📊 Sobre el dataset")
        st.markdown(
            "`BankMarketing.csv` contiene contactos telefónicos de una campaña bancaria. "
            "Cada fila es un cliente y se registran:\n"
            "- **Perfil:** edad, trabajo, estado civil, educación.\n"
            "- **Situación financiera:** mora, crédito hipotecario y personal.\n"
            "- **Gestión comercial:** canal, mes, día, duración, n° de contactos.\n"
            "- **Contexto macroeconómico:** empleo, precios, confianza, euribor.\n"
            "- **Resultado `y`:** `yes` si aceptó la campaña, `no` si no."
        )
        st.subheader("🛠️ Tecnologías utilizadas")
        st.markdown(
            "`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Streamlit`"
        )

    st.info("👈 Usa el menú lateral para navegar. Empieza por **Carga del dataset**.")


# ======================================================================
# MÓDULO 2: CARGA DEL DATASET
# ======================================================================
def modulo_carga():
    st.title("📂 Carga del dataset")
    st.markdown("Sube el archivo **BankMarketing.csv** para habilitar el análisis.")

    archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            df = cargar_csv(archivo)
            ok, mensaje = validar_dataset(df)
        except Exception as e:  # archivo corrupto o no legible
            ok, mensaje, df = False, f"No se pudo leer el archivo: {e}", None

        if ok:
            st.session_state["df"] = df
            st.session_state["nombre_archivo"] = archivo.name
            st.success(f"✅ Archivo '{archivo.name}' cargado correctamente.")
        else:
            st.session_state.pop("df", None)
            st.error(f"❌ {mensaje}")

    df = st.session_state.get("df")
    if df is None:
        st.warning("Aún no hay un dataset cargado. Ningún análisis estará disponible hasta subirlo.")
        return

    if archivo is None:
        st.info(f"Dataset en memoria: **{st.session_state.get('nombre_archivo', '')}**")
        if st.button("🗑️ Descartar dataset"):
            st.session_state.pop("df", None)
            st.rerun()

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{df.shape[0]:,}")
    c2.metric("Columnas", df.shape[1])
    c3.metric("Aceptaron (yes)", f"{(df['y'] == 'yes').sum():,}")

    n = st.slider("Filas a previsualizar", 5, 50, 10)
    st.subheader("Vista previa (head)")
    st.dataframe(df.head(n))


def obtener_dataset():
    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ Primero debes cargar el dataset en el módulo **📂 Carga del dataset**.")
        st.stop()
    return df


# ======================================================================
# MÓDULO 3: EDA (10 ÍTEMS, UNO POR TAB)
# ======================================================================
def item_1(an):
    st.subheader("Ítem 1: Información general del dataset")
    st.markdown("Estructura del dataset: tipos de datos, valores no nulos y conteo de nulos.")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("**`.info()`**")
        st.code(an.info_texto())
    with c2:
        st.markdown("**Tipos de datos y nulos por columna**")
        st.dataframe(an.tabla_tipos(), height=500)
    st.metric("Total de valores nulos (NaN)", int(an.df.isna().sum().sum()))
    st.caption("El dataset no tiene NaN, pero ver Ítem 4: hay faltantes codificados como 'unknown'.")


def item_2(an):
    st.subheader("Ítem 2: Clasificación de variables")
    st.markdown(
        "La función `clasificar_variables()` separa las columnas según su tipo de dato "
        "(numéricas vs categóricas)."
    )
    c1, c2 = st.columns(2)
    c1.metric("Variables numéricas", len(an.numericas))
    c2.metric("Variables categóricas", len(an.categoricas))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Numéricas**")
        st.dataframe(an.df[an.numericas].nunique().rename("Valores únicos").to_frame())
    with c2:
        st.markdown("**Categóricas**")
        st.dataframe(an.df[an.categoricas].nunique().rename("Categorías").to_frame())
    st.caption("La variable objetivo `y` es categórica binaria (yes/no).")


def item_3(an):
    st.subheader("Ítem 3: Estadísticas descriptivas")
    st.markdown("Resumen con `.describe()`, más moda y coeficiente de variación.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**`.describe()` (numéricas)**")
        st.dataframe(an.describe_numerico().round(2))
    with c2:
        st.markdown("**Media, mediana, moda y dispersión**")
        st.dataframe(an.resumen_central().round(2))

    if st.checkbox("Mostrar también el resumen de variables categóricas", key="i3_cat"):
        st.dataframe(an.df[an.categoricas].describe())

    st.markdown("**Interpretación por variable**")
    col = st.selectbox("Variable numérica", an.numericas, key="i3_var")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(an.interpretar_distribucion(col))
    with c2:
        mostrar(an.graf_boxplot(col))


def item_4(an):
    st.subheader("Ítem 4: Análisis de valores faltantes")
    st.markdown(
        "Además de los NaN, en este dataset los faltantes suelen venir codificados como "
        "`'unknown'`."
    )
    incluir = st.checkbox("Considerar 'unknown' como faltante", value=True, key="i4_unk")
    c1, c2 = st.columns([1.1, 1])
    tabla = an.tabla_faltantes()
    with c1:
        st.dataframe(tabla.round(2), height=500)
    with c2:
        mostrar(an.graf_faltantes(incluir))

    total_nan = int(tabla["Nulos (NaN)"].sum())
    con_unk = tabla[tabla["'unknown'"] > 0]["% faltante"].sort_values(ascending=False)
    st.markdown("**Discusión**")
    if con_unk.empty:
        st.markdown(f"No hay NaN ({total_nan}) ni valores 'unknown' en los datos filtrados.")
    else:
        top = con_unk.index[0]
        st.markdown(
            f"- NaN reales: **{total_nan}**.\n"
            f"- Variables con 'unknown': **{len(con_unk)}**; la más afectada es "
            f"**{top}** con **{con_unk.iloc[0]:.1f}%** de sus registros.\n"
            f"- 'unknown' puede ser una categoría informativa (el cliente no quiso o no "
            f"pudo declarar), por lo que conviene mantenerla en el análisis en lugar de eliminarla."
        )


def item_5(an):
    st.subheader("Ítem 5: Distribución de variables numéricas")
    st.markdown("Histograma y boxplot para identificar forma, sesgo y valores atípicos.")
    c1, c2, c3 = st.columns(3)
    with c1:
        col = st.selectbox("Variable", an.numericas, key="i5_col")
    with c2:
        bins = st.slider("Número de bins", 5, 100, 30, key="i5_bins")
    with c3:
        kde = st.checkbox("Mostrar curva KDE", value=True, key="i5_kde")

    c1, c2 = st.columns(2)
    with c1:
        mostrar(an.graf_histograma(col, bins, kde))
    with c2:
        mostrar(an.graf_boxplot(col))
    st.markdown("**Interpretación visual**")
    st.markdown(an.interpretar_distribucion(col))


def item_6(an):
    st.subheader("Ítem 6: Análisis de variables categóricas")
    st.markdown("Conteos, proporciones y gráficos de barras.")
    c1, c2 = st.columns([1, 2])
    with c1:
        col = st.selectbox("Variable categórica", an.categoricas, key="i6_col")
        prop = st.checkbox("Mostrar proporciones (%)", key="i6_prop")
    conteo = an.df[col].value_counts()
    tabla = pd.DataFrame({
        "Conteo": conteo,
        "Proporción (%)": (conteo / conteo.sum() * 100).round(2),
    })
    with c1:
        st.dataframe(tabla)
    with c2:
        mostrar(an.graf_conteo(col, prop))
    moda = conteo.index[0]
    st.markdown(
        f"La categoría más frecuente (moda) de **{col}** es **'{moda}'**, con "
        f"{tabla.loc[moda, 'Proporción (%)']:.1f}% de los registros."
    )


def item_7(an):
    st.subheader("Ítem 7: Análisis bivariado (numérico vs categórico)")
    st.markdown("Comparación de una variable numérica entre quienes aceptan y rechazan (`y`).")
    c1, c2 = st.columns(2)
    with c1:
        idx = an.numericas.index("duration") if "duration" in an.numericas else 0
        col = st.selectbox("Variable numérica", an.numericas, index=idx, key="i7_col")
    with c2:
        outliers = st.checkbox("Mostrar outliers en el boxplot", value=True, key="i7_out")

    c1, c2 = st.columns(2)
    with c1:
        mostrar(an.graf_box_por_y(col, outliers))
    with c2:
        mostrar(an.graf_hist_por_y(col))
    tabla = an.tabla_por_y(col)
    st.dataframe(tabla.round(2))
    if {"yes", "no"} <= set(tabla.index):
        dif = tabla.loc["yes", "median"] - tabla.loc["no", "median"]
        st.markdown(
            f"En **{col}**, la mediana de quienes aceptan es **{tabla.loc['yes', 'median']:.1f}** "
            f"vs **{tabla.loc['no', 'median']:.1f}** de quienes rechazan "
            f"(diferencia de {dif:+.1f})."
        )


def item_8(an):
    st.subheader("Ítem 8: Análisis bivariado (categórico vs categórico)")
    st.markdown("Tasa de aceptación por categoría: ¿qué segmentos responden mejor?")
    idx = an.cat_features.index("education") if "education" in an.cat_features else 0
    col = st.selectbox("Variable categórica", an.cat_features, index=idx, key="i8_col")

    c1, c2 = st.columns(2)
    with c1:
        mostrar(an.graf_apiladas(col))
    with c2:
        tabla = an.tasa_conversion(an.df[col]).reindex(an._orden(col))
        mostrar(an.graf_tasa(tabla, f"Aceptación (%) por {col}"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tabla de contingencia (conteos)**")
        st.dataframe(an.crosstab_y(col))
    with c2:
        st.markdown("**Tasa de aceptación por categoría**")
        st.dataframe(tabla.round(2))
    mejor = tabla["tasa"].idxmax()
    st.markdown(
        f"En **{col}**, el segmento con mayor aceptación es **'{mejor}'** "
        f"({tabla.loc[mejor, 'tasa']:.1f}% sobre {int(tabla.loc[mejor, 'contactos']):,} contactos)."
    )


def item_9(an):
    st.subheader("Ítem 9: Análisis basado en parámetros seleccionados")
    st.markdown("Construye tu propio análisis eligiendo las columnas y el estadístico.")

    st.markdown("#### A. Comparación de grupos")
    c1, c2, c3 = st.columns(3)
    with c1:
        idx = an.cat_features.index("contact") if "contact" in an.cat_features else 0
        grupo = st.selectbox("Agrupar por", an.cat_features, index=idx, key="i9_grupo")
    with c2:
        defecto = [c for c in ["duration", "age"] if c in an.numericas]
        metricas = st.multiselect("Variables numéricas", an.numericas, default=defecto, key="i9_met")
    with c3:
        estad = st.selectbox("Estadístico", ["mean", "median", "std", "min", "max"], key="i9_est")
    ordenar = st.checkbox("Ordenar de mayor a menor (según la primera variable)", key="i9_ord")

    if not metricas:
        st.info("Selecciona al menos una variable numérica.")
    else:
        fig, tabla = an.graf_grupos(grupo, metricas, estad, ordenar)
        mostrar(fig)
        st.dataframe(tabla.round(2))

    st.divider()
    st.markdown("#### B. Relación entre dos variables numéricas")
    c1, c2, c3 = st.columns(3)
    with c1:
        ix = an.numericas.index("duration") if "duration" in an.numericas else 0
        x = st.selectbox("Eje X", an.numericas, index=ix, key="i9_x")
    with c2:
        iy = an.numericas.index("age") if "age" in an.numericas else 0
        y = st.selectbox("Eje Y", an.numericas, index=iy, key="i9_y")
    with c3:
        maximo = min(10000, len(an.df))
        if maximo > 200:
            n = st.slider("Tamaño de la muestra", 200, maximo, min(2000, maximo), key="i9_n")
        else:
            n = maximo
    mostrar(an.graf_dispersion(x, y, n))

    st.divider()
    st.markdown("#### C. Matriz de correlación personalizada")
    cols = st.multiselect("Variables a correlacionar", an.numericas,
                          default=[c for c in an.numericas if c != "pdays"], key="i9_corr")
    if len(cols) >= 2:
        mostrar(an.graf_correlacion(cols))
    else:
        st.info("Selecciona al menos 2 variables.")


def item_10(an):
    st.subheader("Ítem 10: Hallazgos clave")
    st.markdown(
        "Panel resumen: la línea roja punteada es la efectividad global; las barras azules "
        "la superan y las grises quedan por debajo."
    )
    mostrar(an.graf_resumen())
    st.markdown("**Insights principales**")
    for texto in generar_insights(an):
        st.markdown(f"- {texto}")


def modulo_eda(df):
    st.title("🔎 Análisis Exploratorio de Datos (EDA)")

    # ----- Filtros globales en el sidebar -----
    st.sidebar.divider()
    st.sidebar.markdown("### Filtros del EDA")
    edad_min, edad_max = int(df["age"].min()), int(df["age"].max())
    rango_edad = st.sidebar.slider("Rango de edad", edad_min, edad_max, (edad_min, edad_max))
    canales = sorted(df["contact"].unique())
    canales_sel = st.sidebar.multiselect("Canal de contacto", canales, default=canales)
    aplicar = st.sidebar.checkbox("Aplicar filtros", value=True)

    if aplicar:
        datos = df[df["age"].between(*rango_edad) & df["contact"].isin(canales_sel)]
    else:
        datos = df

    if datos.empty:
        st.error("Los filtros seleccionados no devuelven registros. Ajusta los filtros del sidebar.")
        st.stop()

    an = DataAnalyzer(datos)
    st.sidebar.caption(f"Registros analizados: {len(datos):,} de {len(df):,}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(datos):,}")
    c2.metric("Aceptaron", f"{int(an.y_bin.sum()):,}")
    c3.metric("Efectividad", f"{an.tasa_global():.1f}%")
    c4.metric("Variables", datos.shape[1])

    tabs = st.tabs([
        "1️⃣ Info general", "2️⃣ Variables", "3️⃣ Descriptivas", "4️⃣ Faltantes",
        "5️⃣ Numéricas", "6️⃣ Categóricas", "7️⃣ Num vs y", "8️⃣ Cat vs y",
        "9️⃣ Interactivo", "🔟 Hallazgos",
    ])
    items = [item_1, item_2, item_3, item_4, item_5, item_6, item_7, item_8, item_9, item_10]
    for tab, item in zip(tabs, items):
        with tab:
            item(an)


# ======================================================================
# MÓDULO 4: CONCLUSIONES
# ======================================================================
def modulo_conclusiones(df):
    st.title("✅ Conclusiones finales")
    st.markdown("Cinco conclusiones orientadas a la **toma de decisiones** (calculadas con el dataset completo).")
    an = DataAnalyzer(df)

    contacto = an.tasa_conversion(df["contact"])
    previo = an.tasa_conversion(df["poutcome"])
    camp = an.tasa_conversion(an.segmentar_campaign())
    mes = an.tasa_conversion(df["month"])
    edad = an.tasa_conversion(an.segmentar_edad())
    med = df.groupby("y")["duration"].median()

    peso_success = valor(previo, "success", "contactos") / len(df) * 100
    peso_mayo = valor(mes, "may", "contactos") / len(df) * 100

    st.markdown(f"### 1. El canal importa: priorizar celular\n"
                f"El canal *cellular* alcanza **{valor(contacto, 'cellular'):.1f}%** de aceptación frente a "
                f"**{valor(contacto, 'telephone'):.1f}%** del teléfono fijo (global: {an.tasa_global():.1f}%). "
                f"**Decisión:** redirigir la gestión hacia celular siempre que exista el número.")
    st.markdown(f"### 2. Los clientes con éxito previo son la mejor oportunidad\n"
                f"Quienes aceptaron la campaña anterior vuelven a aceptar el **{valor(previo, 'success'):.1f}%** "
                f"de las veces, versus {valor(previo, 'nonexistent'):.1f}% de quienes nunca fueron contactados; "
                f"sin embargo, son solo el {peso_success:.1f}% de la base. "
                f"**Decisión:** priorizar su recontacto y ampliar la base de clientes con historial positivo.")
    st.markdown(f"### 3. Insistir más no vende más\n"
                f"La aceptación cae de **{camp['tasa'].iloc[0]:.1f}%** con un solo contacto a "
                f"**{camp['tasa'].iloc[-1]:.1f}%** con más de 10 intentos. "
                f"**Decisión:** fijar un tope de intentos por cliente y reasignar ese esfuerzo a prospectos nuevos.")
    st.markdown(f"### 4. La conversación de calidad se nota en la duración\n"
                f"La mediana de duración es **{med['yes']:.0f}s** en quienes aceptan vs **{med['no']:.0f}s** en "
                f"quienes rechazan. Esta variable se conoce al terminar la llamada, así que no sirve para "
                f"seleccionar a quién llamar, pero sí para evaluar el guion y la capacitación: "
                f"**Decisión:** medir y reforzar las prácticas de los ejecutivos que logran conversaciones más largas.")
    st.markdown(f"### 5. La estacionalidad y el segmento por edad condicionan la efectividad\n"
                f"Mayo concentra el **{peso_mayo:.0f}%** de los contactos pero convierte solo "
                f"**{valor(mes, 'may'):.1f}%**, mientras que meses de baja carga como marzo "
                f"({valor(mes, 'mar'):.1f}%) o septiembre ({valor(mes, 'sep'):.1f}%) superan el 40%. "
                f"Además, los clientes de 65+ años aceptan **{valor(edad, '>65'):.1f}%** y los de 36-45 solo "
                f"**{valor(edad, '36-45'):.1f}%**. "
                f"**Decisión:** redistribuir la carga de llamadas a lo largo del año y diseñar ofertas específicas por edad.")

    with st.expander("⚠️ Limitaciones a tener en cuenta"):
        st.markdown(
            "- Es un análisis exploratorio: muestra **asociaciones**, no causalidad.\n"
            "- Los meses de alta conversión tienen pocos contactos (posible sesgo de selección).\n"
            "- La base está desbalanceada (~89% 'no'), por lo que conviene comparar tasas y no solo conteos."
        )


# ======================================================================
# PROGRAMA PRINCIPAL
# ======================================================================
def main():
    st.sidebar.title("🏦 Bank Marketing")
    opcion = st.sidebar.radio("Navegación", MENU)

    if "df" in st.session_state:
        st.sidebar.success("Dataset cargado ✅")
    else:
        st.sidebar.warning("Dataset no cargado")

    if opcion == MENU[0]:
        modulo_home()
    elif opcion == MENU[1]:
        modulo_carga()
    elif opcion == MENU[2]:
        modulo_eda(obtener_dataset())
    else:
        modulo_conclusiones(obtener_dataset())


if __name__ == "__main__":
    main()

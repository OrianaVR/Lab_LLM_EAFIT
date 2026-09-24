
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tiktoken

from groq import Groq

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="LLM & NLP Laboratory",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 LLM & NLP Laboratory")

st.markdown(
    """
    Plataforma interactiva para explorar:

    - Modelos de lenguaje (LLM)
    - Tokens y Token IDs
    - Bag of Words
    - TF-IDF
    - Métricas de similitud
    - Embeddings
    - Generación de texto con Groq
    """
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalize_text(text):
    """Normaliza texto para algunas operaciones de NLP."""
    return re.sub(r"\s+", " ", text.strip())


def jaccard_similarity(text1, text2):
    """Calcula similitud de Jaccard entre conjuntos de palabras."""
    words1 = set(re.findall(r"\b\w+\b", text1.lower()))
    words2 = set(re.findall(r"\b\w+\b", text2.lower()))

    union = words1 | words2

    if not union:
        return 0.0

    return len(words1 & words2) / len(union)


@st.cache_resource
def load_embedding_model():
    """
    Carga un modelo de embeddings local.
    Se descarga la primera vez que se utiliza.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def get_tokenizer():
    """
    Tokenizador aproximado compatible con cl100k_base.
    No representa necesariamente el tokenizador interno
    de cada modelo disponible en Groq.
    """
    return tiktoken.get_encoding("cl100k_base")


def tokenize_text(text):
    """Devuelve tokens e IDs."""
    encoding = get_tokenizer()

    token_ids = encoding.encode(text)

    tokens = [
        encoding.decode([token_id])
        for token_id in token_ids
    ]

    return tokens, token_ids


# ============================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================

st.sidebar.header("⚙️ Configuración")

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Introduce tu clave de Groq. No la compartas públicamente."
)

# También permite usar la variable de entorno GROQ_API_KEY.
if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if api_key:
    client = Groq(api_key=api_key)
else:
    client = None


# ============================================================
# MODELOS GROQ
# ============================================================

MODEL_OPTIONS = {
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "GPT-OSS 20B": "openai/gpt-oss-20b",
    "GPT-OSS 120B": "openai/gpt-oss-120b"
}

selected_model_name = st.sidebar.selectbox(
    "Modelo",
    list(MODEL_OPTIONS.keys())
)

selected_model = MODEL_OPTIONS[selected_model_name]

temperature = st.sidebar.slider(
    "Temperatura",
    min_value=0.0,
    max_value=2.0,
    value=0.7,
    step=0.1
)

max_tokens = st.sidebar.number_input(
    "Máximo de tokens de salida",
    min_value=1,
    max_value=8192,
    value=500,
    step=100
)

top_p = st.sidebar.slider(
    "Top P",
    min_value=0.05,
    max_value=1.0,
    value=1.0,
    step=0.05
)

st.sidebar.caption(
    "Los modelos disponibles y sus límites pueden cambiar."
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "🤖 Generación",
    "🔤 Tokens",
    "📚 Bag of Words",
    "📐 Similitud",
    "🔢 Embeddings",
    "⚖️ Comparar modelos"
])


# ============================================================
# 1. GENERACIÓN DE TEXTO
# ============================================================

with tabs[0]:

    st.header("🤖 Generación de texto con Groq")

    st.write(f"Modelo seleccionado: **{selected_model_name}**")

    system_prompt = st.text_area(
        "Instrucciones del sistema",
        value="Eres un asistente educativo experto en NLP y LLM.",
        height=100
    )

    prompt = st.text_area(
        "Escribe tu prompt",
        placeholder="Explica qué es un embedding...",
        height=150
    )

    generate_button = st.button(
        "🚀 Generar texto",
        type="primary",
        key="generate_text"
    )

    if generate_button:

        if not api_key:
            st.error("Introduce tu API Key de Groq.")

        elif not prompt.strip():
            st.warning("Escribe un prompt antes de generar.")

        else:

            try:

                with st.spinner("Generando respuesta..."):

                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]

                    response = client.chat.completions.create(
                        model=selected_model,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=int(max_tokens),
                        top_p=top_p
                    )

                answer = response.choices[0].message.content

                st.subheader("Respuesta")

                st.write(answer)

                st.subheader("Información de uso")

                usage = response.usage

                if usage:

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Tokens de entrada",
                        usage.prompt_tokens
                    )

                    col2.metric(
                        "Tokens de salida",
                        usage.completion_tokens
                    )

                    col3.metric(
                        "Total de tokens",
                        usage.total_tokens
                    )

            except Exception as error:

                st.error(
                    f"Error al generar texto: {error}"
                )


# ============================================================
# 2. TOKENS Y TOKEN IDS
# ============================================================

with tabs[1]:

    st.header("🔤 Tokens y Token IDs")

    st.info(
        "Cada color representa un token diferente. "
        "Debajo de cada token se muestra su Token ID."
    )

    token_text = st.text_area(
        "Escribe el texto que quieres tokenizar:",
        value="Los modelos de lenguaje procesan texto mediante tokens.",
        height=120
    )

    if st.button("🔍 Tokenizar texto"):

        if not token_text.strip():

            st.warning("Introduce un texto.")

        else:

            try:

                token_data = get_token_data(token_text)

                # ------------------------------------------------
                # INFORMACIÓN GENERAL
                # ------------------------------------------------

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "🔢 Cantidad de tokens",
                        len(token_data)
                    )

                with col2:
                    st.metric(
                        "📝 Caracteres",
                        len(token_text)
                    )

                # ------------------------------------------------
                # TOKENS EN COLORES
                # ------------------------------------------------

                st.subheader("🎨 Visualización de tokens")

                colors = [
                    "#FFB6C1",
                    "#ADD8E6",
                    "#98FB98",
                    "#FFD700",
                    "#DDA0DD",
                    "#FFA07A",
                    "#87CEEB",
                    "#F0E68C",
                    "#DA70D6",
                    "#90EE90"
                ]

                html_tokens = ""

                for i, row in token_data.iterrows():

                    token = row["Token"]
                    token_id = row["Token ID"]

                    # Reemplazar espacios para visualización
                    display_token = token.replace(
                        " ",
                        "␠"
                    ).replace(
                        "\n",
                        "↵"
                    )

                    color = colors[
                        i % len(colors)
                    ]

                    html_tokens += f"""
                    <div style="
                        display:inline-block;
                        margin:6px;
                        text-align:center;
                        vertical-align:top;
                    ">

                        <div style="
                            background-color:{color};
                            border:2px solid #333;
                            border-radius:10px;
                            padding:10px 14px;
                            min-width:50px;
                            font-weight:bold;
                            font-size:16px;
                            color:#222;
                        ">
                            {display_token}
                        </div>

                        <div style="
                            margin-top:4px;
                            font-size:12px;
                            color:#555;
                        ">
                            ID: {token_id}
                        </div>

                        <div style="
                            font-size:11px;
                            color:#888;
                        ">
                            #{i}
                        </div>

                    </div>
                    """

                st.markdown(
                    f"""
                    <div style="
                        padding:20px;
                        border-radius:12px;
                        border:1px solid #ddd;
                        background-color:#fafafa;
                        line-height:2.5;
                    ">
                        {html_tokens}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ------------------------------------------------
                # TABLA
                # ------------------------------------------------

                st.subheader("📋 Información detallada")

                st.dataframe(
                    token_data,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # GRÁFICA TOKEN ID
                # ------------------------------------------------

                st.subheader("📊 Token IDs")

                st.bar_chart(
                    token_data.set_index(
                        "Posición"
                    )["Token ID"]
                )

            except Exception as e:

                st.error(
                    f"Error de tokenización: {e}"
                )


# ============================================================
# 3. BAG OF WORDS
# ============================================================

with tabs[2]:

    st.header("📚 Bag of Words")

    st.write(
        "Convierte los documentos en una matriz de frecuencias."
    )

    documents_input = st.text_area(
        "Escribe varios documentos, uno por línea",
        value=(
            "Python es un lenguaje de programación\n"
            "Python permite desarrollar aplicaciones\n"
            "Los modelos de lenguaje utilizan Python"
        ),
        height=150
    )

    if st.button("📊 Construir Bag of Words", key="bow"):

        documents = [
            normalize_text(doc)
            for doc in documents_input.splitlines()
            if normalize_text(doc)
        ]

        if not documents:

            st.warning("Ingresa al menos un documento.")

        else:

            try:

                vectorizer = CountVectorizer(
                    lowercase=True
                )

                matrix = vectorizer.fit_transform(
                    documents
                )

                vocabulary = vectorizer.get_feature_names_out()

                bow_df = pd.DataFrame(
                    matrix.toarray(),
                    columns=vocabulary
                )

                bow_df.index = [
                    f"Documento {i + 1}"
                    for i in range(len(documents))
                ]

                st.subheader("Vocabulario")

                st.write(list(vocabulary))

                st.subheader("Matriz Bag of Words")

                st.dataframe(
                    bow_df,
                    use_container_width=True
                )

                frequencies = bow_df.sum().sort_values(
                    ascending=False
                )

                st.subheader("Frecuencia de palabras")

                frequency_df = frequencies.reset_index()

                frequency_df.columns = [
                    "Palabra",
                    "Frecuencia"
                ]

                fig = px.bar(
                    frequency_df,
                    x="Palabra",
                    y="Frecuencia",
                    title="Frecuencia de palabras"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as error:

                st.error(
                    f"Error en Bag of Words: {error}"
                )


# ============================================================
# 4. SIMILITUD
# ============================================================

with tabs[3]:

    st.header("📐 Métricas de similitud")

    text1 = st.text_area(
        "Texto 1",
        value="Python es un lenguaje de programación",
        height=100
    )

    text2 = st.text_area(
        "Texto 2",
        value="Python permite programar aplicaciones",
        height=100
    )

    similarity_method = st.selectbox(
        "Métrica",
        [
            "Similitud coseno (TF-IDF)",
            "Similitud de Jaccard"
        ]
    )

    if st.button(
        "📐 Calcular similitud",
        key="similarity"
    ):

        if not text1.strip() or not text2.strip():

            st.warning("Ingresa ambos textos.")

        else:

            try:

                if similarity_method == "Similitud coseno (TF-IDF)":

                    vectorizer = TfidfVectorizer()

                    vectors = vectorizer.fit_transform([
                        text1,
                        text2
                    ])

                    score = cosine_similarity(
                        vectors[0:1],
                        vectors[1:2]
                    )[0][0]

                else:

                    score = jaccard_similarity(
                        text1,
                        text2
                    )

                st.metric(
                    "Resultado",
                    f"{score:.4f}"
                )

                st.progress(
                    min(max(float(score), 0.0), 1.0)
                )

                st.caption(
                    "Un valor cercano a 1 indica mayor similitud "
                    "según la representación utilizada."
                )

            except Exception as error:

                st.error(
                    f"Error en similitud: {error}"
                )


# ============================================================
# 5. EMBEDDINGS
# ============================================================

with tabs[4]:

    st.header("🔢 Embeddings")

    st.write(
        "Representación vectorial generada localmente "
        "con Sentence Transformers."
    )

    embedding_text = st.text_area(
        "Texto para embedding",
        value="Machine Learning es una rama de la inteligencia artificial.",
        height=120
    )

    if st.button(
        "🧠 Generar embedding",
        key="generate_embedding"
    ):

        if not embedding_text.strip():

            st.warning("Escribe un texto.")

        else:

            try:

                with st.spinner(
                    "Cargando modelo de embeddings..."
                ):

                    embedding_model = load_embedding_model()

                vector = embedding_model.encode(
                    embedding_text,
                    normalize_embeddings=True
                )

                vector = np.asarray(vector)

                st.metric(
                    "Dimensiones del embedding",
                    len(vector)
                )

                st.subheader(
                    "Primeros 20 valores del vector"
                )

                preview_size = min(20, len(vector))

                embedding_df = pd.DataFrame({
                    "Dimensión": range(preview_size),
                    "Valor": vector[:preview_size]
                })

                st.dataframe(
                    embedding_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader(
                    "Distribución de los valores"
                )

                distribution_df = pd.DataFrame({
                    "Dimensión": range(len(vector)),
                    "Valor": vector
                })

                fig = px.line(
                    distribution_df,
                    x="Dimensión",
                    y="Valor",
                    title="Vector de embedding"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as error:

                st.error(
                    f"Error al generar embedding: {error}"
                )


# ============================================================
# 6. COMPARACIÓN DE MODELOS
# ============================================================

with tabs[5]:

    st.header("⚖️ Comparación de modelos")

    comparison_prompt = st.text_area(
        "Prompt para comparar modelos",
        value="Explica qué es inteligencia artificial en 3 frases.",
        height=120
    )

    available_comparison_models = list(
        MODEL_OPTIONS.items()
    )

    selected_comparison_names = st.multiselect(
        "Selecciona los modelos",
        options=list(MODEL_OPTIONS.keys()),
        default=[
            "Llama 3.1 8B Instant",
            "Llama 3.3 70B Versatile"
        ]
    )

    if st.button(
        "🔄 Comparar respuestas",
        key="compare_models"
    ):

        if not api_key:

            st.error("Introduce tu API Key de Groq.")

        elif not comparison_prompt.strip():

            st.warning("Escribe un prompt.")

        elif not selected_comparison_names:

            st.warning("Selecciona al menos un modelo.")

        else:

            results = []

            for model_name in selected_comparison_names:

                model_id = MODEL_OPTIONS[model_name]

                try:

                    with st.spinner(
                        f"Consultando {model_name}..."
                    ):

                        response = client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {
                                    "role": "user",
                                    "content": comparison_prompt
                                }
                            ],
                            temperature=temperature,
                            max_completion_tokens=int(max_tokens),
                            top_p=top_p
                        )

                    answer = response.choices[0].message.content

                    usage = response.usage

                    input_tokens = (
                        usage.prompt_tokens
                        if usage else None
                    )

                    output_tokens = (
                        usage.completion_tokens
                        if usage else None
                    )

                    results.append({
                        "Modelo": model_name,
                        "Respuesta": answer,
                        "Tokens de entrada": input_tokens,
                        "Tokens de salida": output_tokens
                    })

                except Exception as error:

                    results.append({
                        "Modelo": model_name,
                        "Respuesta": f"Error: {error}",
                        "Tokens de entrada": None,
                        "Tokens de salida": None
                    })

            if results:

                for result in results:

                    st.subheader(
                        result["Modelo"]
                    )

                    st.write(
                        result["Respuesta"]
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Tokens de entrada",
                        result["Tokens de entrada"]
                    )

                    col2.metric(
                        "Tokens de salida",
                        result["Tokens de salida"]
                    )

                    st.divider()


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "LLM & NLP Laboratory | Python + Streamlit + Groq"
)

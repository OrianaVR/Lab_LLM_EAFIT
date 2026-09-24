import streamlit as st
import pandas as pd
import numpy as np
import tiktoken

from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="LLM & NLP Laboratory",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 LLM & NLP Laboratory")
st.write(
    "Plataforma interactiva para experimentar con LLM, "
    "tokens, Bag of Words, similitud, embeddings y generación de texto."
)


# ============================================================
# OPENAI
# ============================================================

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Configuración del modelo")

models = [
    "gpt-5.6-mini",
    "gpt-5.6",
    "gpt-5-mini",
]

model = st.sidebar.selectbox(
    "Modelo GPT",
    models
)

temperature = st.sidebar.slider(
    "Temperatura",
    min_value=0.0,
    max_value=2.0,
    value=0.7,
    step=0.1
)

max_tokens = st.sidebar.number_input(
    "Máximo de tokens",
    min_value=1,
    max_value=4096,
    value=500
)

top_p = st.sidebar.slider(
    "Top P",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "🤖 Generación",
    "🔤 Tokens",
    "📚 Bag of Words",
    "📐 Similitud",
    "🔢 Embeddings"
])


# ============================================================
# GENERACIÓN DE TEXTO
# ============================================================

with tabs[0]:

    st.header("🤖 Generación de texto")

    prompt = st.text_area(
        "Escribe tu prompt",
        height=150,
        placeholder="Explica qué es Machine Learning..."
    )

    if st.button("Generar texto", type="primary"):

        if not api_key:
            st.error("Introduce tu API Key de OpenAI.")
        elif not prompt:
            st.warning("Escribe un prompt.")
        else:

            try:

                response = client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    top_p=top_p
                )

                st.subheader("Respuesta")

                st.write(response.output_text)

            except Exception as e:

                st.error(f"Error: {e}")


# ============================================================
# TOKENS
# ============================================================

with tabs[1]:

    st.header("🔤 Tokenización")

    text = st.text_area(
        "Texto para tokenizar",
        "Los modelos de lenguaje procesan texto mediante tokens."
    )

    try:

        encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)

        decoded_tokens = [
            encoding.decode([token])
            for token in tokens
        ]

        data = pd.DataFrame({
            "Posición": range(len(tokens)),
            "Token": decoded_tokens,
            "Token ID": tokens
        })

        st.metric(
            "Cantidad de tokens",
            len(tokens)
        )

        st.dataframe(
            data,
            use_container_width=True
        )

    except Exception as e:

        st.error(e)


# ============================================================
# BAG OF WORDS
# ============================================================

with tabs[2]:

    st.header("📚 Bag of Words")

    documents = st.text_area(
        "Escribe varios documentos, uno por línea",
        """Python es un lenguaje de programación
Python permite desarrollar aplicaciones
Los modelos de lenguaje utilizan Python"""
    )

    docs = [
        doc.strip()
        for doc in documents.split("\n")
        if doc.strip()
    ]

    if docs:

        vectorizer = CountVectorizer()

        matrix = vectorizer.fit_transform(docs)

        vocabulary = vectorizer.get_feature_names_out()

        df = pd.DataFrame(
            matrix.toarray(),
            columns=vocabulary
        )

        st.subheader("Matriz Bag of Words")

        st.dataframe(
            df,
            use_container_width=True
        )

        st.subheader("Frecuencia de palabras")

        frequencies = df.sum().sort_values(
            ascending=False
        )

        st.bar_chart(frequencies)


# ============================================================
# SIMILITUD
# ============================================================

with tabs[3]:

    st.header("📐 Métricas de similitud")

    text1 = st.text_area(
        "Texto 1",
        "Python es un lenguaje de programación"
    )

    text2 = st.text_area(
        "Texto 2",
        "Python permite programar aplicaciones"
    )

    if st.button("Calcular similitud"):

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform([
            text1,
            text2
        ])

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        st.metric(
            "Similitud coseno",
            f"{similarity:.4f}"
        )

        st.progress(float(similarity))


# ============================================================
# EMBEDDINGS
# ============================================================

with tabs[4]:

    st.header("🔢 Embeddings")

    embedding_text = st.text_area(
        "Texto para obtener embedding",
        "Machine Learning es una rama de la inteligencia artificial."
    )

    if st.button("Generar embedding"):

        if not api_key:

            st.error(
                "Introduce tu API Key de OpenAI."
            )

        else:

            try:

                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=embedding_text
                )

                vector = response.data[0].embedding

                st.metric(
                    "Dimensiones",
                    len(vector)
                )

                st.write("Primeros valores del vector:")

                st.dataframe(
                    pd.DataFrame({
                        "Dimensión": range(20),
                        "Valor": vector[:20]
                    })
                )

                st.write(
                    "Los embeddings representan "
                    "el significado del texto mediante "
                    "un vector numérico."
                )

            except Exception as e:

                st.error(f"Error: {e}")

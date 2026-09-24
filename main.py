import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import textstat
import tiktoken


st.set_page_config(
    page_title="LLM Lab - Groq + OCR",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .token {
        display:inline-block;
        padding:4px 7px;
        margin:3px;
        border-radius:6px;
        font-family:monospace;
        font-size:13px;
        color:white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


@st.cache_resource
def get_ocr():
    """Carga RapidOCR solo cuando el OCR es usado."""
    try:
        from rapidocr import RapidOCR
        return RapidOCR()
    except Exception as exc:
        raise RuntimeError(
            "No se pudo cargar RapidOCR. "
            "Verifica que la app esté usando Python 3.12 "
            "y las dependencias de requirements.txt."
        ) from exc


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_available_models(client):
    try:
        response = client.models.list()
        ids = [m.id for m in response.data]

        excluded = (
            "whisper",
            "guard",
            "tts",
            "speech",
            "orpheus",
            "compound",
        )

        models = [
            model_id
            for model_id in ids
            if not any(word in model_id.lower() for word in excluded)
        ]

        preferred = [m for m in DEFAULT_MODELS if m in models]
        others = sorted(m for m in models if m not in preferred)

        return preferred + others
    except Exception:
        return DEFAULT_MODELS


def extract_text_from_image(uploaded_file):
    """
    OCR con RapidOCR.
    RapidOCR actual devuelve un objeto con:
    result.txts y result.scores.
    """
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    ocr = get_ocr()
    result = ocr(image_np)

    texts = list(getattr(result, "txts", ()) or ())
    scores = list(getattr(result, "scores", ()) or ())

    details = [
        {
            "texto": text,
            "confianza": round(float(score), 4),
        }
        for text, score in zip(texts, scores)
    ]

    return "\n".join(texts), details


def generate_text(
    client,
    model,
    prompt,
    temperature,
    top_p,
    max_tokens,
    style,
):
    if style == "Formal":
        system = (
            "Eres un asistente especializado en redacción formal. "
            "Amplía y organiza la información con lenguaje claro, "
            "preciso y profesional."
        )
    elif style == "Técnica":
        system = (
            "Eres un asistente técnico. Amplía la información usando "
            "terminología técnica cuando sea apropiado, manteniendo "
            "la explicación comprensible."
        )
    else:
        system = (
            "Eres un asistente de lenguaje. Amplía la información "
            "de forma clara, coherente y bien estructurada."
        )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,
    )

    answer = completion.choices[0].message.content or ""
    usage = getattr(completion, "usage", None)

    return answer, {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def bow_table(text):
    words = re.findall(r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b", text.lower())

    if not words:
        return pd.DataFrame(columns=["palabra", "frecuencia"])

    counter = Counter(words)

    return (
        pd.DataFrame(counter.items(), columns=["palabra", "frecuencia"])
        .sort_values("frecuencia", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )


def cosine_text_similarity(text_a, text_b):
    if not text_a.strip() or not text_b.strip():
        return 0.0

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([text_a, text_b])

    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def embedding_similarity(text_a, text_b):
    if not text_a.strip() or not text_b.strip():
        return 0.0

    model = get_embedding_model()
    vectors = model.encode([text_a, text_b])

    return float(
        cosine_similarity(
            vectors[0].reshape(1, -1),
            vectors[1].reshape(1, -1),
        )[0][0]
    )


def text_metrics(text, reference=""):
    words = re.findall(r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b", text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    word_count = len(words)
    sentence_count = len(sentences)

    avg_sentence_length = (
        word_count / sentence_count if sentence_count else 0
    )

    punctuation_count = len(re.findall(r"[.,;:!?]", text))
    punctuation_ratio = (
        punctuation_count / word_count if word_count else 0
    )

    unique_words = len(set(word.lower() for word in words))
    lexical_diversity = (
        unique_words / word_count if word_count else 0
    )

    coherence = min(
        100,
        max(
            0,
            55
            + lexical_diversity * 35
            - abs(avg_sentence_length - 20) * 1.5,
        ),
    )

    syntax = min(
        100,
        max(
            0,
            60
            + punctuation_ratio * 250
            - max(0, avg_sentence_length - 35) * 1.2,
        ),
    )

    grammar = min(
        100,
        max(
            0,
            72
            - len(re.findall(r"\s{2,}", text)) * 5
            - len(re.findall(r"[!?]{2,}", text)) * 5,
        ),
    )

    semantic = (
        cosine_text_similarity(text, reference) * 100
        if reference.strip()
        else 0
    )

    flesch = (
        textstat.flesch_reading_ease(text)
        if text.strip()
        else 0
    )

    return {
        "Palabras": word_count,
        "Oraciones": sentence_count,
        "Longitud promedio de oración": round(avg_sentence_length, 2),
        "Diversidad léxica": round(lexical_diversity, 3),
        "Coherencia estimada": round(coherence, 2),
        "Semántica / similitud": round(semantic, 2),
        "Sintaxis estimada": round(syntax, 2),
        "Gramática estimada": round(grammar, 2),
        "Legibilidad Flesch": round(flesch, 2),
    }


def token_visualization(text):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        token_ids = encoding.encode(text)
        token_texts = [
            encoding.decode([token_id])
            for token_id in token_ids
        ]
        return token_texts, token_ids
    except Exception as exc:
        raise RuntimeError(
            f"No fue posible tokenizar el texto: {exc}"
        ) from exc


def token_html(tokens, token_ids):
    if not tokens:
        return "<p>No hay tokens.</p>"

    html = ""

    for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
        hue = (i * 47) % 360

        safe_token = (
            token.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ", "␠")
            .replace("\n", "↵")
        )

        html += (
            f'<span class="token" '
            f'style="background:hsl({hue},65%,42%)">'
            f"{safe_token} <small>#{token_id}</small>"
            "</span>"
        )

    return html


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Configuración")

groq_key = st.sidebar.text_input(
    "API Key de Groq",
    type="password",
    placeholder="gsk_...",
)

if not groq_key:
    st.title("🤖 LLM Lab — Groq + OCR")
    st.info("Ingresa tu API Key de Groq para comenzar.")
    st.markdown(
        """
        ### Funcionalidades

        Generación de texto, parámetros del LLM, OCR, tokens,
        Token IDs, Bag of Words, similitud, embeddings y métricas.
        """
    )
    st.stop()

client = Groq(api_key=groq_key)
models = get_available_models(client)

model = st.sidebar.selectbox(
    "Modelo LLM",
    models,
)

temperature = st.sidebar.slider(
    "Temperatura",
    0.0,
    2.0,
    0.7,
    0.1,
)

top_p = st.sidebar.slider(
    "Top-P",
    0.1,
    1.0,
    0.95,
    0.05,
)

max_tokens = st.sidebar.slider(
    "Máximo de tokens de salida",
    128,
    8192,
    1024,
    128,
)

style = st.sidebar.radio(
    "Tipo de respuesta",
    ["General", "Formal", "Técnica"],
)


# ============================================================
# TABS
# ============================================================

st.title("🤖 LLM Lab — Plataforma de Lenguaje + OCR")
st.caption(
    "Experimentación con LLM, tokens, Bag of Words, "
    "similitud, embeddings y OCR."
)

tabs = st.tabs(
    [
        "💬 Generación",
        "🖼️ OCR + LLM",
        "🔤 Tokens",
        "📚 Bag of Words",
        "📐 Similitud",
        "🧠 Embeddings",
        "📊 Métricas",
    ]
)


with tabs[0]:
    st.header("Generación de texto")

    prompt = st.text_area(
        "Prompt",
        height=180,
        placeholder="Escribe aquí la instrucción para el modelo...",
    )

    if st.button("🚀 Generar respuesta", type="primary"):
        if not prompt.strip():
            st.warning("Escribe un prompt antes de generar.")
        else:
            try:
                with st.spinner("Generando..."):
                    answer, usage = generate_text(
                        client,
                        model,
                        prompt,
                        temperature,
                        top_p,
                        max_tokens,
                        style,
                    )

                st.subheader("Respuesta")
                st.write(answer)

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Prompt tokens",
                    usage["prompt_tokens"]
                    if usage["prompt_tokens"] is not None
                    else "N/D",
                )
                c2.metric(
                    "Completion tokens",
                    usage["completion_tokens"]
                    if usage["completion_tokens"] is not None
                    else "N/D",
                )
                c3.metric(
                    "Total tokens",
                    usage["total_tokens"]
                    if usage["total_tokens"] is not None
                    else "N/D",
                )

                st.session_state["generated_text"] = answer
                st.session_state["source_text"] = prompt

            except Exception as exc:
                st.error(f"Error al consultar Groq: {exc}")


with tabs[1]:
    st.header("🖼️ OCR + ampliación con LLM")

    uploaded = st.file_uploader(
        "Sube una imagen",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded:
        image = Image.open(uploaded)

        st.image(
            image,
            caption="Imagen cargada",
            use_container_width=True,
        )

        if st.button("🔎 Extraer texto con OCR"):
            try:
                with st.spinner("Analizando imagen..."):
                    extracted, details = extract_text_from_image(
                        uploaded
                    )

                st.session_state["ocr_text"] = extracted
                st.session_state["ocr_details"] = details

            except Exception as exc:
                st.error(f"Error en OCR: {exc}")

    if "ocr_text" in st.session_state:
        extracted = st.session_state["ocr_text"]

        st.subheader("Texto extraído")

        editable_text = st.text_area(
            "Puedes editar el texto antes de enviarlo al LLM.",
            value=extracted,
            height=220,
            key="ocr_editable",
        )

        if st.button("✨ Ampliar texto con LLM", type="primary"):
            instruction = f"""
A partir del siguiente texto extraído de una imagen:

--- TEXTO ---
{editable_text}
--- FIN DEL TEXTO ---

Amplía, organiza y explica la información.
Mantén las ideas principales y agrega contexto útil.
"""

            try:
                with st.spinner("Generando ampliación..."):
                    answer, usage = generate_text(
                        client,
                        model,
                        instruction,
                        temperature,
                        top_p,
                        max_tokens,
                        style,
                    )

                st.subheader("Respuesta ampliada")
                st.write(answer)

                st.session_state["generated_text"] = answer
                st.session_state["source_text"] = editable_text

                if usage["total_tokens"] is not None:
                    st.caption(
                        f"Tokens utilizados: {usage['total_tokens']}"
                    )

            except Exception as exc:
                st.error(
                    f"Error al generar la ampliación: {exc}"
                )

        if st.session_state.get("ocr_details"):
            st.subheader("Confianza del OCR")
            st.dataframe(
                pd.DataFrame(st.session_state["ocr_details"]),
                use_container_width=True,
            )


with tabs[2]:
    st.header("🔤 Tokens y Token IDs")

    text_for_tokens = st.text_area(
        "Texto para tokenizar",
        value=st.session_state.get(
            "generated_text",
            st.session_state.get("ocr_text", ""),
        ),
        height=180,
    )

    if st.button("Tokenizar"):
        try:
            tokens, token_ids = token_visualization(
                text_for_tokens
            )

            st.write(f"Cantidad de tokens: **{len(tokens)}**")
            st.markdown(
                token_html(tokens, token_ids),
                unsafe_allow_html=True,
            )

            st.dataframe(
                pd.DataFrame(
                    {
                        "posición": range(len(tokens)),
                        "token": tokens,
                        "token_id": token_ids,
                    }
                ),
                use_container_width=True,
            )

            st.info(
                "Se usa cl100k_base como referencia didáctica; "
                "el tokenizer exacto puede variar según el modelo."
            )

        except Exception as exc:
            st.error(str(exc))


with tabs[3]:
    st.header("📚 Bag of Words")

    bow_text = st.text_area(
        "Texto para analizar",
        value=st.session_state.get(
            "generated_text",
            st.session_state.get("ocr_text", ""),
        ),
        height=180,
    )

    if bow_text.strip():
        bow = bow_table(bow_text)

        st.dataframe(bow, use_container_width=True)

        if not bow.empty:
            st.bar_chart(
                bow.set_index("palabra")["frecuencia"].head(15)
            )


with tabs[4]:
    st.header("📐 Métricas de similitud")

    source = st.text_area(
        "Texto original / OCR",
        value=st.session_state.get("source_text", ""),
        height=150,
    )

    generated = st.text_area(
        "Texto generado",
        value=st.session_state.get("generated_text", ""),
        height=150,
    )

    if st.button("Calcular similitud"):
        if source.strip() and generated.strip():
            similarity = cosine_text_similarity(
                source,
                generated,
            )

            st.metric(
                "Similitud TF-IDF / coseno",
                f"{similarity * 100:.2f}%",
            )
            st.progress(min(1.0, similarity))
        else:
            st.warning(
                "Necesitas un texto original y uno generado."
            )


with tabs[5]:
    st.header("🧠 Embeddings")

    embedding_a = st.text_area(
        "Texto A",
        value=st.session_state.get("source_text", ""),
        height=130,
    )

    embedding_b = st.text_area(
        "Texto B",
        value=st.session_state.get("generated_text", ""),
        height=130,
    )

    if st.button("Calcular embeddings"):
        if embedding_a.strip() and embedding_b.strip():
            try:
                with st.spinner("Calculando embeddings..."):
                    similarity = embedding_similarity(
                        embedding_a,
                        embedding_b,
                    )

                    vectors = get_embedding_model().encode(
                        [embedding_a, embedding_b]
                    )

                st.metric(
                    "Similitud semántica",
                    f"{similarity * 100:.2f}%",
                )

                st.write(
                    f"Dimensión del embedding: "
                    f"**{vectors.shape[1]}**"
                )

                st.dataframe(
                    pd.DataFrame(
                        vectors,
                        index=["Texto A", "Texto B"],
                    ).iloc[:, :20],
                    use_container_width=True,
                )

            except Exception as exc:
                st.error(f"Error calculando embeddings: {exc}")
        else:
            st.warning("Ingresa ambos textos.")


with tabs[6]:
    st.header("📊 Métricas del texto generado")

    metric_text = st.text_area(
        "Texto a evaluar",
        value=st.session_state.get("generated_text", ""),
        height=220,
    )

    reference = st.text_area(
        "Texto de referencia",
        value=st.session_state.get("source_text", ""),
        height=150,
    )

    if st.button("📊 Analizar texto"):
        if not metric_text.strip():
            st.warning("Escribe o genera un texto primero.")
        else:
            metrics = text_metrics(
                metric_text,
                reference,
            )

            cols = st.columns(4)

            cols[0].metric("Palabras", metrics["Palabras"])
            cols[1].metric("Oraciones", metrics["Oraciones"])
            cols[2].metric(
                "Coherencia estimada",
                f'{metrics["Coherencia estimada"]:.1f}/100',
            )
            cols[3].metric(
                "Gramática estimada",
                f'{metrics["Gramática estimada"]:.1f}/100',
            )

            st.dataframe(
                pd.DataFrame(
                    {
                        "Métrica": list(metrics.keys()),
                        "Valor": list(metrics.values()),
                    }
                ),
                use_container_width=True,
            )

            st.info(
                "Coherencia, sintaxis y gramática son indicadores "
                "heurísticos y no sustituyen una evaluación "
                "lingüística profesional."
            )


st.divider()

st.caption(
    "LLM Lab — Plataforma educativa de experimentación "
    "con LLM, OCR, tokens, similitud y embeddings."
)

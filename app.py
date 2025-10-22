import os
import re
import streamlit as st
from dotenv import load_dotenv

from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage

try:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY bulunamadı. Lütfen .env veya Streamlit Secrets ayarını kontrol edin.")
        st.stop()
except Exception as e:
    st.error(f"Ortam değişkenleri yüklenirken hata oluştu: {e}")
    st.stop()

@st.cache_resource
def load_and_prepare_anayasa():
    try:
        with st.spinner("Anayasa txt'i işleniyor..."):
            txt_path = "anayasa.txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
            else:
                st.error(f"{txt_path} bulunamadı.")
                return None

            # "Madde X –" desenine göre bölme
            madde_metinleri = re.split(r'(Madde \d+ –)', full_text)
            anayasa_docs = []

            for i in range(1, len(madde_metinleri), 2):
                madde_basligi = madde_metinleri[i].strip()
                madde_icerigi = madde_metinleri[i+1].strip()
                content = f"{madde_basligi}\n{madde_icerigi}"

                anayasa_docs.append(
                    Document(
                        content=content,
                        meta={"kaynak": "anayasa.txt", "madde": madde_basligi}
                    )
                )

            return anayasa_docs
    except Exception as e:
        st.error(f"Anayasa txt yüklenirken hata oluştu: {e}")
        return None

@st.cache_resource(show_spinner="🧠 Anayasa veritabanı hazırlanıyor...")
def create_inmemory_index(anayasa_docs):
    if not anayasa_docs:
        return None
    
    try:
        document_store = InMemoryDocumentStore()

        embedder = SentenceTransformersDocumentEmbedder(
            model="trmteb/turkish-embedding-model"
        )

        pipeline = Pipeline()
        pipeline.add_component("embedder", embedder)
        pipeline.add_component("writer", DocumentWriter(document_store=document_store))
        pipeline.connect("embedder.documents", "writer.documents")

        pipeline.run({"embedder": {"documents": anayasa_docs}})
        return document_store
    except Exception as e:
        st.error(f"Vektör veritabanı oluşturulamadı: {e}")
        return None

def build_anayasa_rag(document_store):
    if not document_store:
        return None

    try:
        retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=7)
        text_embedder = SentenceTransformersTextEmbedder(model="trmteb/turkish-embedding-model")

        template = [
            ChatMessage.from_system(
                "Sen, yalnızca sağlanan Türkiye Anayasası metinlerini kullanarak soruları yanıtlayan bir asistansın."
                "Cevabını oluşturmak için SADECE aşağıda verilen belgeleri kullan."
                "Eğer bilgi belgelerde yoksa, 'Bu bilgi sağlanan Anayasa metninde bulunmamaktadır.' de."
                "Cevabını dört cümleyle sınırla ve mümkünse madde numarasını belirt."
            ),
            ChatMessage.from_user(
        """Belgeler:
        {% for doc in documents %}
          {{ doc.content }}
        {% endfor %}

        Soru: {{question}}
        Yanıt:
        """
        ),
        ]

        prompt_builder = ChatPromptBuilder(template=template,  required_variables={"question"})
        generator = GoogleGenAIChatGenerator(
            model="gemini-2.0-flash",
            api_key=Secret.from_token(GOOGLE_API_KEY),
            generation_kwargs={
                "temperature": 0.4,
                "top_p": 0.95
            }
        )

        rag = Pipeline()
        rag.add_component("text_embedder", text_embedder)
        rag.add_component("retriever", retriever)
        rag.add_component("prompt_builder", prompt_builder)
        rag.add_component("generator", generator)

        rag.connect("text_embedder.embedding", "retriever.query_embedding")
        rag.connect("retriever.documents", "prompt_builder.documents")
        rag.connect("prompt_builder.prompt", "generator.messages")

        return rag
    except Exception as e:
        st.error(f"RAG pipeline oluşturulamadı: {e}")
        return None

def main():
    st.set_page_config(page_title="Türkiye Anayasası Asistanı", page_icon="⚖️")
    st.title("⚖️ Türkiye Anayasası Asistanı")
    st.caption("Anayasa maddelerine dayalı akıllı soru-cevap sistemi")

    anayasa_docs = load_and_prepare_anayasa()
    if anayasa_docs:
        document_store = create_inmemory_index(anayasa_docs)
        if document_store:
            rag_pipeline = build_anayasa_rag(document_store)
        else:
            rag_pipeline = None
    else:
        rag_pipeline = None

    if not rag_pipeline:
        st.warning("Sistem başlatılamadı. Lütfen hata mesajlarını kontrol edin.")
        st.stop()

    if "chat" not in st.session_state:
        st.session_state.chat = []

    st.subheader("💡 Örnek Sorular")
    col1, col2, col3, col4 = st.columns(4)
    
    example_questions = [
        "Cumhurbaşkanı nasıl seçilir?",
        "Anayasa nedir?",
        "Temel hak ve özgürlükler nelerdir?",
        "Anayasa nasıl değiştirilir?"
    ]
    
    question_clicked = None
    with col1:
        if st.button("🏛️ " + example_questions[0], use_container_width=True):
            question_clicked = example_questions[0]
    with col2:
        if st.button("📜 " + example_questions[1], use_container_width=True):
            question_clicked = example_questions[1]
    with col3:
        if st.button("⚖️ " + example_questions[2], use_container_width=True):
            question_clicked = example_questions[2]
    with col4:
        if st.button("✏️ " + example_questions[3], use_container_width=True):
            question_clicked = example_questions[3]
    
    st.markdown("---")

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = None
    
    manual_input = st.chat_input("Anayasa chatbotuna dilediğinizi sorun...")
    
    if question_clicked:
        user_input = question_clicked
    elif manual_input:
        user_input = manual_input

    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Maddeler taranıyor..."):
            try:
                result = rag_pipeline.run({
                    "text_embedder": {"text": user_input},
                    "prompt_builder": {"question": user_input}
                })
                chat_message = result["generator"]["replies"][0]
                response = chat_message._content[0].text if chat_message._content else "Yanıt alınamadı."
            except Exception as e:
                response = f"Hata: {e}"

        st.session_state.chat.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
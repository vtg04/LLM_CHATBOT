import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi
import fitz
import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.text for p in soup.find_all('p')])
    return text

urls = ["https://www.apple.com/apple-vision-pro/", "https://support.apple.com/en-in/guide/apple-vision-pro/welcome/visionos","https://support.apple.com/guide/apple-vision-pro/read-pdfs-dev449021435/visionos","https://www.apple.com/apple-vision-pro/specs/","https://support.apple.com/en-in/117810","https://support.apple.com/en-in/guide/apple-vision-pro/tan489cfe222/visionos"]
web_texts = [scrape_website(url) for url in urls]

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

pdf_paths = ["Apple_Vision.pdf", "Apple_PER.pdf"]
pdf_texts = [extract_text_from_pdf(pdf_path) for pdf_path in pdf_paths]

video_ids = ["TX9qSaGXFyg", "Vb0dG-2huJE"]
youtube_texts = []

for video_id in video_ids:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    youtube_texts.append(' '.join([entry['text'] for entry in transcript]))

all_texts = pdf_texts + web_texts + youtube_texts

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "faiss_index.idx")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model_name = "CallComply/Starling-LM-11B-alpha" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

index = faiss.read_index("faiss_index.idx")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_most_relevant_document(query, index, model):
    query_embedding = embedding_model.encode([query])
    _, I = index.search(np.array(query_embedding), 1)
    return all_texts[I[0][0]]

def generate_response(query):
    relevant_document = get_most_relevant_document(query, index, embedding_model)
    prompt = f"Question: {query}\nContext: {relevant_document}\nAnswer:"
    
    response = llama_pipeline(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text']

query = "What is the privacy policy of Apple Vision Pro?"
response = generate_response(query)
print(response)

import streamlit as st
import pandas as pd
import time

st.title("Apple Vision Pro Chatbot")

st.markdown("""
<style>
.chat-container {
    background-color: #f1f1f1;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.user-message {
    text-align: right;
    font-size: 16px;
}
.bot-message {
    text-align: left;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="input")
    submit = st.form_submit_button("Send")

if submit and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = generate_response(user_input)
    st.session_state.messages.append({"role": "bot", "content": response})

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-container user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-container bot-message">{message["content"]}</div>', unsafe_allow_html=True)

if submit:
    time.sleep(1) 
    st.experimental_rerun()

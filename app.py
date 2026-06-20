import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Sentiment Analysis App", page_icon="🤖")

@st.cache_resource
def load_assets():
    model = load_model('sentiment_lstm_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

try:
    model, tokenizer = load_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False

st.title("📝 Text Sentiment Analysis (LSTM)")
st.write("Masukkan teks ulasan (review) untuk mengetahui apakah sentimennya **Positif** atau **Negatif**.")

if not model_loaded:
    st.error("Model belum dilatih! Jalankan `python train.py` terlebih dahulu di terminal Anda.")
else:
    user_input = st.text_area("Masukkan ulasan di bawah ini:", placeholder="Contoh: Makanannya enak tapi pelayanannya lambat...")

    if st.button("Prediksi Sentimen"):
        if user_input.strip() == "":
            st.warning("Teks tidak boleh kosong ya!")
        else:
            with st.spinner('Menganalisis...'):
                seq = tokenizer.texts_to_sequences([user_input])
                # Maxlen disesuaikan menjadi 50 agar sinkron dengan model baru
                padded = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
                
                prediction = model.predict(padded)[0][0]
                
                st.divider()
                if prediction > 0.5:
                    st.success(f"✨ **Sentimen: POSITIF**")
                    st.write(f"Skor Kepercayaan: {prediction:.2%}")
                else:
                    st.error(f"💔 **Sentimen: NEGATIF**")
                    st.write(f"Skor Kepercayaan: {(1 - prediction):.2%}")
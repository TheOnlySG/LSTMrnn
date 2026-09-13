import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import numpy as np
import streamlit as st

#loading model
model = load_model('lstmrnn.h5')

#loading tokenizer
with open('tokenizer.json' , 'r') as file:
    tokenizer_json = file.read()
tokenizer = tokenizer_from_json(tokenizer_json)


def predict_next_word(model , tokenizer , input_text , max_sequence_len):
    token_input = tokenizer.texts_to_sequences([input_text])[0]

    if len(token_input) >= max_sequence_len:
        token_input = token_input[-(max_sequence_len-1):]

    token_input = pad_sequences([token_input] , maxlen=max_sequence_len-1,padding='pre')
    predicted = model.predict(token_input,verbose=0)
    predicted_index = np.argmax(predicted , axis=1)[0]

    return tokenizer.index_word.get(predicted_index)

    # return None



#streamlit
st.title("Next Word Prediction with LSTM")
input_text = st.text_input("Enter the sentence", "to be or not to be")
if st.button("predict next word"):
    max_seq_len = model.input_shape[1]+1
    next_word = predict_next_word(model , tokenizer , input_text , max_seq_len)
    st.write(f'next word : {next_word}')

    

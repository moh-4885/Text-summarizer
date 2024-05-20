from tensorflow.keras.models import load_model
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from attention import AttentionLayer
import pickle

nltk.download('stopwords')

encoder_model = load_model('encoder_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
decoder_model= load_model('decoder_model.h5', custom_objects={'AttentionLayer': AttentionLayer})

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}
stop_words = set(stopwords.words('english'))

max_text_len=30
max_summary_len=8
with open('tokenizer.pickle', 'rb') as tokenizer_file:
    x_tokenizer = pickle.load(tokenizer_file)
with open('y_tokenizer.pickle', 'rb') as tokenizer_file:
    y_tokenizer = pickle.load(tokenizer_file)
    
reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index


class Summarizer():
    
    def __init__(self):
        pass
    def text_cleaner(self,text,num):
        newString = text.lower()
        newString = BeautifulSoup(newString, "lxml").text
        newString = re.sub(r'\([^)]*\)', '', newString)
        newString = re.sub('"','', newString)
        newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
        newString = re.sub(r"'s\b","",newString)
        newString = re.sub("[^a-zA-Z]", " ", newString)
        newString = re.sub('[m]{2,}', 'mm', newString)
        if(num==0):
            tokens = [w for w in newString.split() if not w in stop_words]
        else:
            tokens=newString.split()
        long_words=[]
        for i in tokens:
            if len(i)>1:                                                 #removing short word
                long_words.append(i)
        return (" ".join(long_words)).strip()
    def vectorisation(self,text):
        text = self.text_cleaner(text,0)
        
        text_seq=x_tokenizer.texts_to_sequences(np.array([text]))
        vector=pad_sequences(text_seq,  maxlen=max_text_len, padding='post')
        return vector.reshape(1,max_text_len)


    def decode_sequence(self,text):
        # Encode the input as state vectors.
        input_seq = self.vectorisation(text)
        e_out, e_h, e_c = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1,1))

        # Populate the first word of target sequence with the start word.
        target_seq[0, 0] = target_word_index['sostok']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:

            output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index[sampled_token_index]

            if(sampled_token!='eostok'):
                decoded_sentence += ' '+sampled_token

            # Exit condition: either hit max length or find stop word.
            if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            e_h, e_c = h, c

        return decoded_sentence




# summary = Summarizer()
# print(summary.vectorisation('salt separate area pain makes hard regulate salt putting like salt go ahead get product'))







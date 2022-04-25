import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from tools import clean_title
import numpy as np

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.train_df = None
        self.test_df = None
        self.train_set = None
        self.test_set = None
    
    def load(self):
        self.train_df = pd.read_csv(f"{self.file_path}/train.csv").head(10000)
        self.test_df  = pd.read_csv(f"{self.file_path}/test.csv").head(10000)

    def preprocess(self, df, is_train=False):
        metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        for similarity in metrics:
            L = []
            for index, row in df.iterrows():
                title1 = clean_title(row['title1_en'])
                title2 = clean_title(row['title2_en'])
                vector = CountVectorizer().fit_transform([title1, title2]).toarray()
                L.append(pairwise_distances(vector, metric=similarity)[0][1]*100)
            df.insert(5, similarity, L, True)
        
        del df['title1_en']
        del df['title2_en']
        X, y = df.loc[:, df.columns != 'label'], df['label']
        if is_train:    
            return train_test_split(X, y, test_size=0.25, random_state=42)

        return X, y
    
    def get_sets(self):
        if self.train_df is None:
            self.load()
        if self.train_set is not None and self.test_set is not None:
            return self.train_set, self.test_set
        
        self.train_set = self.preprocess(self.train_df, True)
        self.test_set = self.preprocess(self.test_df)
        return self.train_set, self.test_df

    def get_train_set(self):
        if self.train_df is None:
            self.load()
        if self.train_set is not None:
            return self.train_set
        self.train_set = self.preprocess(self.train_df, True)
        
        return self.train_set
    
    def get_test_set(self):
        if self.test_df is None:
            self.load()
        if self.test_set is not None:
            return self.test_set
        self.test_set = self.preprocess(self.test_df)
        return self.test_set
    

# import tensorflow as tf

# import tensorflow_hub as hub
# import tensorflow_datasets as tfds
# tfds.disable_progress_bar()

# from official.modeling import tf_utils
# from official import nlp
# from official.nlp import bert

# # Load the required submodules
# import official.nlp.optimization
# import official.nlp.bert.bert_models
# import official.nlp.bert.configs
# import official.nlp.bert.run_classifier
# import official.nlp.bert.tokenization
# import official.nlp.data.classifier_data_lib
# import official.nlp.modeling.losses
# import official.nlp.modeling.models
# import official.nlp.modeling.networks



# class DataLoader_Bert:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.train_df = None
#         self.test_df = None
#         self.train_set = None
#         self.test_set = None
#         self.hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
#         tf.io.gfile.listdir(self.hub_url_bert)
#         self.tokenizer = bert.tokenization.FullTokenizer(vocab_file = os.path.join(self.hub_url_bert, "vocab.txt"), do_lower_case=True)
    
#     def load(self):
#         self.train_df = pd.read_csv(f"{self.file_path}/train.csv").head(10000)
#         self.test_df  = pd.read_csv(f"{self.file_path}/test.csv").head(10000)
    
#     def encode_sentence(self, s):
#         tokens = list(self.tokenizer.tokenize(s.numpy()))
#         tokens.append('[SEP]')
#         return self.tokenizer.convert_tokens_to_ids(tokens)
    
#     def bert_encode(self, df):
#         nbr_data = len(self.train_df.shape[:1])
#         sentence1 = tf.ragged.constant([self.encode_sentence(s, self.tokenizer) for s in np.array(df['title1_en'])])
#         sentence2 = tf.ragged.constant([self.encode_sentence(s, self.tokenizer) for s in np.array(df['title2_en'])])
#         cls = [self.tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
#         input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)
#         input_mask = tf.ones_like(input_word_ids).to_tensor()

#         type_cls = tf.zeros_like(cls)
#         type_s1 = tf.zeros_like(sentence1)
#         type_s2 = tf.ones_like(sentence2)
#         input_type_ids = tf.concat( 
#             [type_cls, type_s1, type_s2], axis=-1).to_tensor()

#         inputs = {
#             'input_word_ids': input_word_ids.to_tensor(),
#             'input_mask': input_mask,
#             'input_type_ids': input_type_ids}

#         return inputs

#     def preprocess(self, df, is_train=False):
       
    
#     def get_sets(self):
#         if self.train_df is None:
#             self.load()
#         if self.train_set is not None and self.test_set is not None:
#             return self.train_set, self.test_set
        
#         self.train_set = self.preprocess(self.train_df, True)
#         self.test_set = self.preprocess(self.test_df)
#         return self.train_set, self.test_df

#     def get_train_set(self):
#         if self.train_df is None:
#             self.load()
#         if self.train_set is not None:
#             return self.train_set
#         self.train_set = self.preprocess(self.train_df, True)
        
#         return self.train_set
    
#     def get_test_set(self):
#         if self.test_df is None:
#             self.load()
#         if self.test_set is not None:
#             return self.test_set
#         self.test_set = self.preprocess(self.test_df)
#         return self.test_set
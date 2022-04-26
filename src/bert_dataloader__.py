
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import tensorflow as tf

import transformers as trsf

class DataTokenizerGen(tf.keras.utils.Sequence):
    
    def __init__(self, data, pairs=None, labels=None, batch_size=128, max_length=256, include_labels=True):
        self.max_length = max_length
        if pairs is None:
            self.pairs = data[["title1_en", "title2_en"]].values.astype("str")
        else:
            self.pairs = pairs
        self.labels = labels
        self.include_labels = include_labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.pairs))
        self.tokenizer = trsf.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
                
    def __len__(self):
        return len(self.pairs) // self.batch_size
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        pairs = self.pairs[indexes]
    
        if self.include_labels:
            labels = np.array(self.labels[indexes], dtype="int32")
            return self.get_features(pairs), labels
        else:
            return self.get_features(pairs)
    
    def get_features(self, pairs):
        encoded = self.tokenizer.batch_encode_plus(
            pairs.tolist(), 
            add_special_tokens=True, 
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )
        
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        return [input_ids, attention_masks, token_type_ids]
       
        

class DataLoader_Bert:
    def __init__(self, file_path, max_length=128, batch_size=64):
        self.file_path = file_path
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.train_gen = None
        self.valid_gen = None
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = DataTokenizerGen
        self.labels = ["disagreed", "agreed", "unrelated"]

    
    def __load(self):
        df = pd.read_csv(f"{self.file_path}/train.csv").head(3000)
        self.train_df, self.valid_df = train_test_split(df, test_size=0.2, random_state=42)
        self.test_df  = pd.read_csv(f"{self.file_path}/test.csv").head(1000)
        
    
    def __preprocess(self, df, is_train=False):
        y = None
        batch_size = 1
        if is_train:
            batch_size = self.batch_size
            labels = df["label"].apply(lambda x: 0 if x == "disagreed" else 1 if x == "agreed" else 2)
            y = tf.keras.utils.to_categorical(labels, num_classes=3)
            return self.tokenizer(df, labels=y, batch_size=batch_size, max_length=self.max_length, include_labels=is_train)
        return self.tokenizer(None, pairs=df, labels=y, batch_size=batch_size, max_length=self.max_length, include_labels=is_train)
        
    def prepare_training(self):
        if self.train_df is None or self.valid_df is None:
            self.__load()
            
        if self.train_gen is not None and self.valid_gen is not None:
            return self.train_gen, self.valid_gen
        
        self.train_gen = self.__preprocess(self.train_df, True)
        self.valid_gen = self.__preprocess(self.valid_df, True)
        return self.train_gen, self.valid_gen

    def prepare_testing(self):
        if self.test_df is None:
            self.__load()
        if "x" in self.test_df.columns:
            return self.test_df
        
        self.test_df["x"] =  self.test_df.apply(lambda row : self.__preprocess([[row["title1_en"], row["title2_en"]]], False), axis = 1)
        return self.test_df
    
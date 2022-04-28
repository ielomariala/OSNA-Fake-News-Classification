
import tensorflow as tf
import transformers as trsf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class BertModel:
    
    def __init__(self, max_length=256) -> None:
        self.strategy = tf.distribute.MirroredStrategy()
        self.model = None
        self.max_length = max_length
        self.history = None
                
    def create_model(self, bert_trainable=False):
        with self.strategy.scope():
            # Encoded token ids from BERT tokenizer.
            input_ids = tf.keras.layers.Input(
                shape=(self.max_length,), dtype=tf.int32, name="input_ids"
            )
            # Attention masks indicates to the model which tokens should be attended to.
            attention_masks = tf.keras.layers.Input(
                shape=(self.max_length,), dtype=tf.int32, name="attention_masks"
            )
            # Token type ids are binary masks identifying different sequences in the model.
            token_type_ids = tf.keras.layers.Input(
                shape=(self.max_length,), dtype=tf.int32, name="token_type_ids"
            )
            # Loading pretrained BERT model.
            bert_model = trsf.TFBertModel.from_pretrained("bert-base-uncased")
            # Freeze the BERT model to reuse the pretrained features without modifying them.
            bert_model.trainable = bert_trainable

            bert_output = bert_model(
                input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
            )
            sequence_output = bert_output.last_hidden_state
            pooled_output = bert_output.pooler_output
            
            # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
            bi_lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)
            )(sequence_output)
            
            # Applying hybrid pooling approach to bi_lstm sequence output.
            avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
            max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
            concat = tf.keras.layers.concatenate([avg_pool, max_pool])
            dropout = tf.keras.layers.Dropout(0.3)(concat)
            output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
            self.model = tf.keras.models.Model(
                inputs=[input_ids, attention_masks, token_type_ids], outputs=output
            )

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss="categorical_crossentropy",
                metrics=["acc"],
            )
            
    def fit(self, train_data, valid_data, epochs=1, batch_size=256):
        checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
                                     save_weights_only=True, save_best_only=True, period=1)
    
        self.history = self.model.fit(train_data,
                                      validation_data=valid_data,
                                      epochs=epochs, 
                                      batch_size=batch_size, 
                                      callbacks=[checkpoint])
        return self.history

    def plot_history(self):
        """Plot loss and accuracy"""
        plt.subplot(211)
        plt.plot(self.history.history["loss"], label="loss")
        plt.plot(self.history.history["val_loss"], label="val_loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='lower right')
        plt.subplot(212)
        plt.plot(self.history.history["acc"], label="acc")
        plt.plot(self.history.history["val_acc"], label="val_acc")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig("history.png")
        plt.show()
        
    def save_weights(self, weights):
        self.model.save_weights(weights)
    
    def load_from_weights(self, weights):
        self.create_model()
        self.model.load_weights(weights)
        
    def get_submission(self, test_data, labels):
        def get_prediction(x):
            result = self.model.predict(x)[0]
            idx = np.argmax(result)
            pred = result[idx]
            return (pred, labels[idx])
        
        submission = pd.DataFrame(columns=["id", "label"])
        for i in range(len(test_data)):
            row = test_data[i]
            submission.loc[i] = row[1], get_prediction(row[0])
        submission.to_csv("predictions.csv", columns=['id','label'], index=False)
        return submission
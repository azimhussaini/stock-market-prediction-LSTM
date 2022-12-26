"""Creating and training model"""

import logging
import datetime as dt
import numpy as np
import time
from utils import LoggingCallback
import tensorflow as tf

class Model():
    """Building and inference models"""
    def __init__(self, training_type, configs):
        self.training_type = training_type
        self.configs = configs

    def build_model(self):
        """
        Build Model based on configuration
        """
        configs = self.configs
        time_step = configs.data_generator['input_width']
        features =  len(configs.data_generator['input_columns'])
        target_cols_len = len(configs.data_generator['label_columns'])
        target_step = configs.data_generator['label_width']
        input = tf.keras.Input(shape=(time_step, features))
        x = input
        model_type = configs.training["model_type"]
        for layer in configs.models[model_type]["layers"]:
            neurons = layer["neurons"] if "neurons" in layer else None
            activation = layer["activation"] if "activation" in layer else None
            return_sequences = layer["return_sequences"] if "return_sequences" in layer else None
            dropout_rate = layer["dropout_rate"] if "dropout_rate" in layer else None
        
            if layer["type"] == "dense":
                x = tf.keras.layers.Dense(neurons, activation=activation)(x)
            if layer["type"] == "flatten":
                x = tf.keras.layers.Flatten()(x)
            if layer["type"] == "reshape":
                target_shape = [configs.data_generator["label_width"], len(configs.data_generator["label_columns"])]
                x = tf.keras.layers.Reshape(target_shape)(x)
            if layer["type"] == "lstm":
                x = tf.keras.layers.LSTM(neurons, return_sequences=return_sequences)(x)
            if layer["type"] == "dropout":
                x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        # Add final dense layer for prediction. The number of neurons depends label_width and number label_columns. 
        if self.training_type == "multi_sequence_oneshot":
            x = tf.keras.layers.Dense(target_step*target_cols_len)(x)
            x = tf.keras.layers.Reshape([target_step, target_cols_len])(x)
        else:
            x = tf.keras.layers.Dense(target_cols_len)(x)
        
        self.model = tf.keras.Model(input, x)
       
        lr = configs.training["learning_rate"]
        if configs.training["optimizer"] == "adam":
            training_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif configs.training["optimizer"] == "sgd":
            training_optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        self.model.compile(loss=configs.training["loss"], 
                            optimizer=training_optimizer)
        logging.info(f"Model built successfully for {self.training_type}")
        

    def compile_model(self):
        lr = self.configs.training["learning_rate"]
        if self.configs.training["optimizer"] == "adam":
            training_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=0.5)
        elif self.configs.training["optimizer"] == "sgd":
            training_optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        self.model.compile(loss=self.configs.training["loss"], 
                            optimizer=training_optimizer)
        logging.info('Model compiled successfully')
    
    
    def train(self, train_data, val_data, epochs, batch_size, save_dir=None):
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6)
        ]
        history = self.model.fit(train_data, 
                                epochs=epochs, 
                                batch_size=batch_size,
                                validation_data = val_data,
                                callbacks=[LoggingCallback(print_fcn=logging.info)]
        )
        logging.info('Training Completed')
        return history

    
    def predict_point_by_point(self, data):
        """Predicting one time step ahead each time given the last sequence of true data"""
        logging.info("Predicting point-by-point...")
        predicted = self.model.predict(data)
        # predicted = np.reshape(predicted, (predicted.size,))
        return predicted


    def predict_multi_seq_oneshot(self, data):
        logging.info("Predicting multiple sequences using one shot model...")
        predicted = self.model.predict(data)
        return predicted


    @tf.function
    def _percentage_change_normalize(self, element, p0):
        # Get Shape
        batch_size = tf.shape(element)[0]
        window_size = tf.shape(element)[1]
        feature_size = tf.shape(element)[2]
        # Get starting value of each window
        # p0 = element[:,0,:]
        # Reshape
        p0 = tf.repeat(p0, repeats=[window_size], axis=0)
        p0 = tf.reshape(p0, shape=(batch_size, window_size, feature_size))
        pc = tf.divide(element, p0) - 1
        return pc
    
    @tf.function
    def _update_input_seq(self, x, logits, p0, n_pred_cols, return_de_normalize_pred=False):
        input_columns = self.configs.data_generator["input_columns"]
        # Update input sequence with prediction
        predicted_value = tf.expand_dims(logits, axis=0)
        predicted_de_normalized = p0[:, :n_pred_cols] * (1+predicted_value)
        updated_input = tf.concat([x[:, 1:, :][:,:,:n_pred_cols], predicted_de_normalized], axis=1)
        # Update prediction dependent features for input
        if set(["MA", "Bollinger_Upper", "Bollinger_Lower"]) <= set(input_columns):
            MA = tf.reduce_mean(updated_input[:, :, :1], axis=1)
            STD = tf.math.reduce_std(updated_input[:, :, :1], axis=1)
            bollinger_upper = MA + (STD*2)
            bollinger_lower = MA - (STD*2)
            new_features = tf.concat([MA, bollinger_upper, bollinger_lower], axis=-1)
            new_features = tf.expand_dims(new_features, axis=0)
            new_input = tf.concat([predicted_de_normalized, new_features], axis=-1)
        elif "MA" in input_columns:
            MA = tf.reduce_mean(updated_input[:, :, :1], axis=1)
            new_features = tf.concat([MA], axis=-1)
            new_features = tf.expand_dims(new_features, axis=0)
            new_input = tf.concat([predicted_de_normalized, new_features], axis=-1)
        else:
            new_input = predicted_de_normalized
        x = tf.concat([x[:, 1:, :], new_input], axis=1)
        if return_de_normalize_pred == True:
            return x, predicted_de_normalized
        else:
            return x

    @tf.function
    def _mult_seq_train(self, x, y, metrics):
        # Call Metrics
        loss_fn = metrics["loss_fn"]
        train_loss = metrics["train_loss"]
        training_optimizer = metrics["training_optimizer"]
        
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss_value = loss_fn(y, logits)
        # Update mean loss
        train_loss.update_state(loss_value)
        # Retrive gradients
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        training_optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return logits


    @tf.function
    def _mult_seq_val(self, x, y, metrics):
        # Call Metrics
        loss_fn = metrics["loss_fn"]
        val_loss = metrics["val_loss"]
        # Run prediction
        logits = self.model(x)
        loss_value = loss_fn(y, logits)
        val_loss.update_state(loss_value)
        
        return logits


    # @tf.function
    def multi_seq_training(self, train_data, val_data, epochs):
        # Setup history dict
        history = {
            'loss': [],
            'val_loss': []
        }

        # Get data variables
        label_width = self.configs.data_generator["label_width"]
        n_pred_cols = len(self.configs.data_generator["label_columns"])
        
        # Initialize Optimizer
        lr = self.configs.training["learning_rate"]
        if self.configs.training["optimizer"] == "adam":
            training_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif self.configs.training["optimizer"] == "sgd":
            training_optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        
        # Initialize Loss
        if self.configs.training["loss"] == "mse":
            loss_fn = tf.keras.losses.MeanSquaredError()
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        val_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

        # Setup metrics dict
        metrics = {
            "training_optimizer": training_optimizer,
            "loss_fn": loss_fn,
            "train_loss": train_loss,
            "val_loss": val_loss
        }

        # Custom Training
        for epoch in range(epochs):
            logging.info(f"\nStart of each epoch {epoch}")
            start_time = time.time()
            for step, (x_batch_train, y_batch_train) in enumerate(train_data):
                for x_batch, y_batch in zip(x_batch_train, y_batch_train):
                    # Set input and labels
                    x_batch = tf.expand_dims(x_batch, axis=0)
                    x = x_batch
                    p0 = x[:,0,:]
                    y = tf.expand_dims(y_batch, axis=0)
                    y_normalize = self._percentage_change_normalize(y, p0[:,:n_pred_cols])
                    for i in range(label_width):
                        x_normalize = self._percentage_change_normalize(x,  p0)
                        # Run Training Step
                        logits = self._mult_seq_train(x_normalize, y_normalize[0][i], metrics)
                        # Update input sequence with prediction
                        x = self._update_input_seq(x, logits, p0, n_pred_cols)

            # logging.info epoch results
            train_loss = metrics['train_loss'].result()
            history['loss'].append(train_loss)
            logging.info(f"Mean Training Loss: {train_loss}")
            
            # Validation Step
            for x_batch_val, y_batch_val in val_data:
                for x_batch, y_batch in zip(x_batch_val, y_batch_val):
                    # Set input and labels
                    x_batch = tf.expand_dims(x_batch, axis=0)
                    x = x_batch
                    p0 = x[:,0,:]
                    y = tf.expand_dims(y_batch, axis=0)
                    y_normalize = self._percentage_change_normalize(y, p0[:,:n_pred_cols])
                    for i in range(label_width):
                        x_normalize = self._percentage_change_normalize(x,  p0)
                        logits = self._mult_seq_val(x_normalize, y_normalize[0][i], metrics)
                        # Update input sequence with prediction
                        x = self._update_input_seq(x, logits, p0, n_pred_cols)

            # # logging.info epoch results
            val_loss = metrics['val_loss'].result()
            history['val_loss'].append(val_loss)
            logging.info(f"Mean Validation Loss: {val_loss}")
            logging.info(f"Time taken: {(time.time() - start_time):.2f}")
            # Reset Metrics
            metrics['train_loss'].reset_states()
            metrics['val_loss'].reset_states()
        
        return history



    def predict_multi_seq(self, data, normalized=True):
        n_pred_cols = len(self.configs.data_generator["label_columns"])
        label_width = self.configs.data_generator["label_width"]
        predict_multi_seq = []
        y_multi_seq = []
        for x_batch_data, y_batch_data in data:
            for x_batch, y_batch in zip(x_batch_data, y_batch_data):
                # Set input and labels
                x_batch = tf.expand_dims(x_batch, axis=0)
                x = x_batch
                p0 = x[:,0,:]
                y = tf.expand_dims(y_batch, axis=0)
                predict_seq = tf.zeros(dtype=tf.float32, shape=(0, n_pred_cols)) 
                for i in range(label_width):
                    x_normalize = self._percentage_change_normalize(x,  p0)
                    logits = self.model(x_normalize)
                    # Update input sequence with prediction
                    if not normalized:
                        x, de_normalized_pred = self._update_input_seq(x, logits, p0, n_pred_cols, return_de_normalize_pred=True)
                        de_normalized_pred = tf.squeeze(de_normalized_pred, axis=0)
                        predict_seq = tf.concat([predict_seq, de_normalized_pred], axis=0)
                    else:
                        x = self._update_input_seq(x, logits, p0, n_pred_cols)
                        predict_seq = tf.concat([predict_seq, logits], axis=0)

                if normalized:
                    y_normalize = self._percentage_change_normalize(y, p0[:,:n_pred_cols])
                    y_normalize = tf.squeeze(y_normalize)
                    y_multi_seq.append(np.array(y_normalize))
                    # y_multi_seq.append(y_normalize)
                else:
                    y = tf.squeeze(y, axis=0)
                    y_multi_seq.append(np.array(y))
                    # y_multi_seq.append(y)
                
                predict_multi_seq.append(np.array(predict_seq))
        
        assert (len(y_multi_seq) == len(predict_multi_seq)), f"Length label sequences: {len(y_multi_seq)} and prediction sequences: {len(predict_multi_seq)} does not match."
        
        y_labels = np.concatenate(y_multi_seq)
        predictions =  np.concatenate(predict_multi_seq)
        

        return y_labels, predictions









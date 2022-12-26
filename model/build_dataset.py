"""Window Generator for sequence dataset"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


class WindowGenerator():
    
    def __init__(self, train_df, test_df, input_width, label_width, strides, batch_size, normalize, label_columns=None):
        # Store Data
        self.train_df = train_df
        self.test_df = test_df

        self.batch_size = batch_size
        self.normalize = normalize

        # label columns indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.strides = strides

        self.total_window_size = input_width + label_width
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]


    def __repr__(self):
        """Printing sample indices of datasets features"""
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label columns: {self.label_columns}'
        ])
    
    def min_max_normalize(self, element):
        # Get Shape
        batch_size = tf.shape(element)[0]
        window_size = tf.shape(element)[1]
        feature_size = tf.shape(element)[2]
        # Calculate Min and Max value for each feature
        min_val = tf.reduce_min(element, axis=1)
        max_val = tf.reduce_max(element, axis=1)
        # Reshape 
        min_val = tf.repeat(min_val, repeats=[window_size], axis=0)
        min_val = tf.reshape(min_val, shape = (batch_size, window_size, feature_size))
        max_val = tf.repeat(max_val, repeats=[window_size], axis=0)
        max_val = tf.reshape(max_val, shape = (batch_size, window_size, feature_size))
        # Calculate MinMax 
        element = tf.divide((element - min_val), (max_val - min_val))
        element = tf.reshape(element, (batch_size, window_size, feature_size))
        return element

    
    def percentage_change_normalize(self, element):
        """
        Normalizing data based on percentage change in price based on starting price of the batch
        
        Input:
            element: (Tensor) Input tensor with size (batch, window, features) 
        
        Return:
            pc: (Tensor) Similar size as input   
        """

        # Get Shape
        batch_size = tf.shape(element)[0]
        window_size = tf.shape(element)[1]
        feature_size = tf.shape(element)[2]
        # Get starting value of each window
        p0 = element[:,0,:]
        # Reshape
        p0 = tf.repeat(p0, repeats=[window_size], axis=0)
        p0 = tf.reshape(p0, shape=(batch_size, window_size, feature_size))
        pc = tf.divide(element, p0) - 1
        return pc


    def de_normalize_percentage_change(self, data, normalize_price, label_name):
        """
        De_normalized percentage change normalization

        Input:
            data: (Numpy Series) Usually the train/test 'Close' or price to get initial price for each batch
            normalized_price: ()
        """
        data = data[label_name]
        label_index = self.label_columns.index(label_name)
        normalize_price = normalize_price[:, label_index]
        normalize_price = normalize_price.reshape((-1,self.label_width))
        p0 = data[:-self.input_width:self.label_width]
        p0 = p0[:normalize_price.shape[0]]
        p0 = np.array(p0)
        p0 = p0.reshape(-1,1)
        de_normalize = np.multiply((1+normalize_price),p0)
        de_normalize = np.reshape(de_normalize, [-1, 1])
        return de_normalize

    def split_window(self, features):
        """Given a list of consecutive inputs, 
        the split_window method will convert them to a 
        window of inputs and a window of labels."""

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], 
                axis=-1
            )
        
        # set shapes
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    
    def plot(self, inputs, labels, plot_col, model=None, max_subplots=3):
        """
        Plot method that allows a simple visualization of the split window
        It aligns inputs, labels, and predictions based on the time that the item refers to
        """

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col}')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index], label="Inputs")

            if self.label_columns:
                label_column_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_column_index = plot_col_index
            
            if label_column_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_column_index], label="Labels", c="green")

            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_column_index], label="Predictions", marker="X", c="red")

            if n==0:
                plt.legend()
            
            plt.xlabel("Time")

    
    def make_dataset(self, data):
        """take a time series DataFrame and convert it to a tf.data.Dataset of (input_window, label_window) pairs 
        using the tf.keras.utils.timeseries_dataset_from_array function"""
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.strides,
            shuffle=False,
            batch_size=self.batch_size
        )

        if self.normalize:
            ds = ds.map(self.percentage_change_normalize)
            # ds = ds.map(self.min_max_normalize)

        
        ds = ds.map(self.split_window)

        return ds
    
    def get_labels(self, dataset):
        labels = []
        target_label_no = len(self.label_columns)
        for element in dataset:
            batch_label = element[1]
            batch_label = tf.reshape(batch_label, shape=[-1, target_label_no]) 
            labels.append(np.array(batch_label))
        labels = np.concatenate(labels)
        return labels

    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get an example of batch of inputs & labels for plotting"""
        result = next(iter(self.train))
        self._example = result
        return result





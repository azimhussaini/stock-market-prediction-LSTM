# Stock Trend Prediction using LSTM

This project is an attempt to model stock price trend for **n** periods in future. In other words, instead of predicting the stock price for next period, the project attempts predicts sequence of next **n** periods given the sequence of input periods.

The model uses two different ways of predicting sequence of prices for upcoming periods.
1. Single Point Prediction, where model is trained to predict next time step
2. Single short predictions, where entire time series is predicted at once
3. Feedback predictions, where model only makes single step prediction and the prediction is fed back as it new input to the input sequence to predict next step predictions. The process continues until the prediction sequence is completed

## Data and Dataset Generator
The project downloads data from **Alpha Vantage** using free API key which can be obtained from **[here](https://www.alphavantage.co/support/#api-key)**. The stock price data is then pass through sequence data generator class to produce train and test datasets. The basic concept for sequence data generator is referred from **[TensorFlow - Time Series Forecasting Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series#multi-step_dense)**. The WindowGenerator class generates consecutive sequences into batches of windows from the training and test data using `tf.data.Dataset`. The class further provides optionality to normalize data using "percentage change" based on starting value of the window. 

<br>

# Example & Usage
1. [Requirements](#requirements)
2. [Directory Setup](#directory-setup)
3. [Configuration](#configuration)
4. [Download Data](#download-data)
5. [Data Preprocessing](#data-preprocessing)
6. [Window Generator](#window-generator)
7. [Model](#model)
8. [Training](#training)

 <br>

## Requirements
Virtual Environment and Python 3 are recommended.

```
# Setting up virtualenv for windows
python -m venv project_env
project_env\Scripts\activate.bat
pip install -r requirements.txt
```

## Directory Setup
```
├── README.md
├── requirement.txt
├── config.json
├── utils.py
├── download_data.py
├── data
│   ├── stock_market_data-XXX.csv
│   ├── train_data
│   │   ├── train_XXX.csv
│   ├── test_data
│   │   ├── test_XXX.csv
├── model
│   ├── build_dataset.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── saved_models
│   │   ├── timeStamp
│   │   │   ├── single_step
│   │   │   ├── multi_step
│   ├── model_logs
├── train.py
├── plotting.py
├── plots
│   ├── learning_curves
│   ├── predictions

```

## Configuration
The entire project is controlled by `config.json` file. This includes **Data** settings, **Model** architecture, **Training** hyperparameter, & **Prediction** types. It is important to review entire set configs parameters before running any scripts. The `config.json` file can be found in project's root directory. 
```json
{
    "data": {
        "api_key": "YTINWG5HI2HG5PXL",
        "ticker": "SPY",
        "start_date": "2016-01-01",
        "train_test_split": 0.2
    },
    "data_generator":{
        "input_width": 10,
        "input_columns": ["Close", "MA", "Bollinger_Upper", "Bollinger_Lower"],
        "strides": 1,
        "label_width": 1,
        "label_columns": ["Close"], 
        "normalize": true
    },
    "training":{
        "epochs": 10,
        "batch_size":32,
        "model_type": "lstm_model",
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.0003,
        "training_type": ["single_point", "multi_sequence_oneshot", "multi_sequence_feedback"],
        "multi_sequence_length": 5,
        "save_model": true
    },
    ...
```

## Download Data
To download data, run `download_data.py`. The script will download data for `config.data["ticker"]` using free Alpha Vantage API key and save the csv file to data directory. It further uses split ratio as defined `config.data["train_test_split"]` to split data into training and test data. The train and test csv files are saved in to its respective folders under data directory.

<br>

## Data Preprocessing
At this moment, the model can handle following calculated features based on  input width of each sequence as defined in `configs.data_generator["input_width"]`.
- Moving Average -> `MA`
- Bollinger Band (Upper) -> `Bollinger_Upper`
- Bollinger Band (Lower) -> `Bollinger_Lower`

It is also possible to create additional indicators by updating the `preprocessing.py` and `model.py`. The `preprocessing.py` scripts creates additional features for entire data at once. If feedback model is used for multi-sequence prediction, where one-step prediction is fed back to the model, additional features should be recalculated for the updated sequence that includes a new prediction point. Such update is made in `model.py` script within `_update_input_seq` method. 

## Window Generator
Once the data is preprocessed, Window Generator class process the data into sequential streams to form tensorflow time series datasets using [tf.keras.utils.timeseries_dataset_from_array](https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array). The parameters for window generator are defined in configuration file under `configs.data_generator`. This includes:
- **input_width**: Time periods for input sequence
- **input_columns**: List of input columns including additional calculated features
- **strides**: Period between successive output sequences. Default setting should be 1
- **label_width**: Length of labels. For one-step model and multi-sequence model using feedback model, default label width should be 1
- **label_columsn**: List of target column name
- **normalize**: Boolean - Whether to normalize or not. The normalization is done using percentage change based on starting value of the input sequence

## Model
The LSTM Model structure is also defined in configuration file under `configs.models["lstm_model"]`. The model structure can be customized by removing/adding layers. At this moment, the model can handle **LSTM** and **Dropout**. layers. Note that the **last** LSTM layer should not return sequence and `return_sequences` parameter should be set to `false`. The model adjust the neurons of final Dense layer depending on training type as defined in `configs.training["training_type"]`.

```json
"models":{
        "lstm_model":{
            "layers":[
                {
                    "type": "lstm",
                    "neurons": 128,
                    "return_sequences": true
                },
                {
                    "type": "dropout",
                    "dropout_rate": 0.2
                },
                {
                    "type": "lstm",
                    "neurons": 64,
                    "return_sequences": true
                },
                {
                    "type": "dropout",
                    "dropout_rate": 0.1
                },
                {
                    "type":"lstm",
                    "neurons": 32,
                    "return_sequences": false
                }
            ]
        }
    }
```


## Training
The model can handle *3* different types of training where each training type has its own use case. It is also possible to run all three types of training and prediction at once. Set training types in configuration file under `configs.training["training_type"]`, that takes list of strings. The training types includes:
- `single_point`: The model trains on true history of input sequence to predict next single time step. Under such training, prediction points are relatively close to actual points. However, since the prediction points have had true prior history behind, the model does not need to know much about the time-series and instead attempt to predicts the points that won't be too far from the last time step. Even though, such training might not be able to provide accurate forecast for next time step price, it does provide useful representation of the **range** the next price point should be. This could be very useful in volatility prediction and anomaly detection.
- `multi_sequence_oneshot`: The model trains on true history of input sequence to predict **n** sequences of time-steps in future. That is instead of predicting next time step, the model attempts to predict **n** time steps. The number of sequence prediction can be defined in configuration file under `configs.training["multi_sequence_length"]`. 
- `multi_sequence_feedback`: The model uses the pre-train `single_point` model to produce single time step output and then feedbacks the output into itself as updated input sequence to output next time step. The process continues until the model produces prediction sequence, as `configs.training["multi_sequence_length"]`, conditioned on previous predictions. Such training is useful in producing trend-like predictions in order to analyze how well the model can predict future trends.

In order to train model based on above configuration as defined in `configs.json`, run `train.py`. 

**Model Logs**: The training logs for each type of training are saved within `model/model_logs` directory. 

**Learning Curves**: The learning curves for each type of training are saved within `plots/learning_curves` directory.

**Prediction Plots**: The predictions for each model are made on test set. The prediction plots for each model are saved within `plot/predictions` directory.


<br>

# Future Work
- Instead of normalizing at every sequence, normalize after n number of sequence to maintain stability during running input sequence.







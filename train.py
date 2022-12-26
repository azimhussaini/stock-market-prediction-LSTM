"""Training model"""

import logging
import os
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import json

from sklearn.metrics import mean_squared_error
import tensorflow as tf

from utils import Params, set_logger, NumpyArrayEncoder
from utils import Params
from model.preprocessing import select_data, create_features
from model.build_dataset import WindowGenerator
from model.model import Model
from plotting import plot_loss, plot_label_prediction

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='model', metavar='',
                    help='Model directory containing params.json')
parser.add_argument('--data_dir', default='data', metavar='',
                    help="Directory containing the dataset")

if __name__ == '__main__':
    # Set random seed to reproduce the graph
    tf.random.set_seed(42)

    # Setup logger
    args = parser.parse_args()
    time_stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    log_path = os.path.join(args.model_dir, "model_logs")
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    set_logger(os.path.join(log_path,time_stamp))

    logging.info("Tensorflow has access to the following devices")
    for device in tf.config.list_physical_devices():
        logging.info(f". {device}")

    # load parameters from json files
    config_path = os.path.join("config.json") 
    assert os.path.isfile(config_path), f"No json config file found at {config_path}"
    configs = Params(config_path)
    logging.info("Configurations successfully loaded")

    # Train and Test data
    train_filename = "train_" + configs.data["ticker"] + ".csv"
    test_filename = "test_" + configs.data["ticker"] + ".csv"
    train_data_path = os.path.join(args.data_dir, "train_data", train_filename)
    assert train_data_path, "Training Data file does not exist"
    test_data_path = os.path.join(args.data_dir, "test_data", test_filename)
    assert test_data_path, "Test Data file does not exist"
    train_df = pd.read_csv(train_data_path)
    logging.info("Training Data successfully loaded")
    test_df = pd.read_csv(test_data_path)
    logging.info("Test Data successfully loaded")

    # Window Generator Configs
    input_width = configs.data_generator["input_width"]
    input_columns = configs.data_generator["input_columns"]
    # shift = configs.data_generator["shift"]
    strides = configs.data_generator["strides"]
    label_width = configs.data_generator["label_width"]
    label_columns = configs.data_generator["label_columns"]
    normalize = configs.data_generator["normalize"]
    start_date = configs.data["start_date"]
    
    # Training Configs
    batch_size = configs.training["batch_size"]
    epochs = configs.training["epochs"]
    lr = configs.training["learning_rate"]
    loss = configs.training["loss"]
    optimizer = configs.training["optimizer"]
    save_model = configs.training["save_model"]

    # Setup Prediction Dictionary
    model_predictions = {}

    # Preprocess
    train_df = create_features(train_df, max_input_width=input_width, input_cols=input_columns)
    test_df = create_features(test_df, max_input_width=input_width, input_cols=input_columns)
    train_df, train_df_date = select_data(train_df, input_columns, unwanted_cols=['Unnamed: 0'], start_date=start_date)
    test_df, test_df_date = select_data(test_df, input_columns, unwanted_cols=['Unnamed: 0'])

    # Initialize Results Dictionary
    model_results = {
        "single_step": {}
    }

    # Generate Dataset
    dataset = WindowGenerator(train_df, test_df, input_width, label_width, strides, batch_size, normalize, label_columns)
    test_labels = dataset.get_labels(dataset.test)

    # Build Single Point Model
    my_model = Model(training_type="single_point", configs=configs)
    my_model.build_model()
    my_model.compile_model()
    
    # Train_model Single Step Model
    logging.info("#######Training Single Step Model#######")
    history = my_model.train(dataset.train, dataset.test, epochs, batch_size)
    train_loss = np.around(history.history["loss"], decimals=5)
    val_loss = np.around(history.history["val_loss"], decimals=5)
    single_step_loss_fig = plot_loss(train_loss, val_loss, "Single Step Model")
    model_results["single_step"].update({"train_loss": train_loss})
    model_results["single_step"].update({"val_loss": val_loss})

    # Single Point Prediction:
    prediction_points = my_model.predict_point_by_point(dataset.test)
    model_results["single_step"].update({"prediction_points": prediction_points})
    MSE_normalized = mean_squared_error(test_labels, prediction_points, multioutput="raw_values")
    for i, label_name in enumerate(label_columns):
        print(f"Mean Squared Error (Normalized) - Single Point Prediction for {label_name}: {MSE_normalized[i]}")
        model_results["single_step"].update({"MSE_Normalized": [label_name, MSE_normalized[i]]})
    
    for i, label_name in enumerate(label_columns):
        de_normalize_predictions_temp = dataset.de_normalize_percentage_change(data=test_df, normalize_price=prediction_points, label_name=label_name)
        de_normalize_labels_temp = dataset.de_normalize_percentage_change(data=test_df, normalize_price=test_labels, label_name=label_name)
        MSE = mean_squared_error(de_normalize_labels_temp, de_normalize_predictions_temp)
        print(f"Mean Squared Error (De-normalized) - Single Point Prediction for {label_name}: {MSE:.2f}")
        model_results["single_step"].update({"MSE_De_Normalized": [label_name, MSE]})
        if i == 0:
            de_normalize_predictions = de_normalize_predictions_temp
            de_normalize_labels = de_normalize_labels_temp
        else:
            de_normalize_predictions = np.concatenate((de_normalize_predictions, de_normalize_predictions_temp), axis=1)
            de_normalize_labels = np.concatenate((de_normalize_labels, de_normalize_labels_temp), axis=1)
            model_results["single_step"].update({"de_normalize_predictions": de_normalize_predictions})


    # Multiple Sequence Training - One Shot Model
    if "multi_sequence_oneshot" in configs.training["training_type"]:
        model_results.update({"multi_sequence_oneshot": {}})
        # Generate Multi Sequence Dataset
        sequence_length = configs.training["multi_sequence_length"]
        configs.data_generator["label_width"] = sequence_length
        label_width = configs.data_generator["label_width"]
        strides = label_width
        dataset_multi = WindowGenerator(train_df, test_df, input_width, label_width, strides, batch_size, normalize, label_columns)
        test_labels_multi = dataset_multi.get_labels(dataset=dataset_multi.test)
        # Training & Prediction
        training_data = dataset_multi.train
        test_data = dataset_multi.test
        logging.info("#######Training Multiple Step Model#######")
        mult_seq_oneshot_model = Model(training_type="multi_sequence_oneshot", configs=configs)
        mult_seq_oneshot_model.build_model()
        mult_seq_oneshot_model.compile_model()
        history_multi_seq_oneshot = mult_seq_oneshot_model.train(training_data, test_data, epochs, batch_size)
        # Loss
        train_loss = np.around(history_multi_seq_oneshot.history["loss"], decimals=5)
        val_loss = np.around(history_multi_seq_oneshot.history["val_loss"], decimals=5)
        multi_seq_loss_fig = plot_loss(train_loss, val_loss, "Multi Sequence Model")
        model_results["multi_sequence_oneshot"].update({"train_loss": train_loss})
        model_results["multi_sequence_oneshot"].update({"val_loss": val_loss})
        # Prediction
        prediction_multi_seq_oneshot = mult_seq_oneshot_model.predict_multi_seq_oneshot(test_data)
        prediction_multi_seq_oneshot = np.reshape(prediction_multi_seq_oneshot, newshape=[-1, len(label_columns)])
        model_results["multi_sequence_oneshot"].update({"prediction_points": prediction_multi_seq_oneshot})
        # MSE
        MSE_normalized = mean_squared_error(test_labels_multi, prediction_multi_seq_oneshot, multioutput="raw_values")
        for i, label_name in enumerate(label_columns):
            print(f"Mean Squared Error (Normalized) - Multi Sequence Prediction for {label_name}: {MSE_normalized[i]}")
            model_results["multi_sequence_oneshot"].update({"MSE_Normalized": [label_name, MSE_normalized[i]]})
        # De Normalized MSE
        for i, label_name in enumerate(label_columns):
            de_normalize_predictions_multi_seq_oneshot_temp = dataset_multi.de_normalize_percentage_change(data=test_df, normalize_price=prediction_multi_seq_oneshot, label_name=label_name)
            de_normalize_labels_multi_seq_oneshot_temp = dataset_multi.de_normalize_percentage_change(data=test_df, normalize_price=test_labels_multi, label_name=label_name)
            MSE = mean_squared_error(de_normalize_labels_multi_seq_oneshot_temp, de_normalize_predictions_multi_seq_oneshot_temp)
            print(f"Mean Squared Error (De-normalized) - Multi Sequence Prediction for {label_name}: {MSE:.2f}")
            model_results["multi_sequence_oneshot"].update({"MSE_De_Normalized": [label_name, MSE]})
            if i == 0:
                de_normalize_predictions_multi_seq_oneshot = de_normalize_predictions_multi_seq_oneshot_temp
                de_normalize_labels_multi_seq_oneshot = de_normalize_labels_multi_seq_oneshot_temp
            else:
                de_normalize_predictions_multi_seq_oneshot = np.concatenate((de_normalize_predictions_multi_seq_oneshot, de_normalize_predictions_multi_seq_oneshot_temp), axis=1)
                de_normalize_labels_multi_seq_oneshot = np.concatenate((de_normalize_labels_multi_seq_oneshot, de_normalize_labels_multi_seq_oneshot_temp), axis=1)
                model_results["multi_sequence_oneshot"].update({"de_normalize_predictions": de_normalize_predictions_multi_seq_oneshot})


    # Multiple Sequence Training using feedback model
    if "multi_sequence_feedback" in configs.training["training_type"]:
        model_results.update({"multi_sequence_feedback": {}})
        sequence_length = configs.training["multi_sequence_length"]
        configs.data_generator["label_width"] = sequence_length
        label_width = configs.data_generator["label_width"]
        strides = label_width
        normalize = configs.data_generator['normalize'] = False
        dataset = WindowGenerator(train_df, test_df, input_width, label_width, strides, batch_size, normalize, label_columns)
        training_data = dataset.train
        test_data = dataset.test
        # mult_seq_feedback_model = Model(training_type="multi_sequence_feedback", configs=configs)
        # mult_seq_feedback_model.build_model()
        # mult_seq_feedback_model.compile_model()
        logging.info("#######Training Feedback Model#######")
        history_mult_seq_feedback = my_model.multi_seq_training(training_data, test_data, epochs)
        # Loss
        train_loss = np.around(history_mult_seq_feedback["loss"], decimals=5)
        val_loss = np.around(history_mult_seq_feedback["val_loss"], decimals=5)
        multi_seq_loss_feedback_fig = plot_loss(train_loss, val_loss, "Multi Sequence Model")
        model_results["multi_sequence_feedback"].update({"train_loss": train_loss})
        model_results["multi_sequence_feedback"].update({"val_loss": val_loss})
        # Multiple sequence prediction
        y_multi_seq, predict_multi_seq = my_model.predict_multi_seq(test_data, normalized=True)
        y_multi_seq_de_norm, predict_multi_seq_de_norm = my_model.predict_multi_seq(test_data, normalized=False)
        model_results["multi_sequence_feedback"].update({"prediction_points": predict_multi_seq})
        model_results["multi_sequence_feedback"].update({"de_normalize_predictions": predict_multi_seq_de_norm})
        # MSE
        MSE_normalized = mean_squared_error(y_multi_seq, predict_multi_seq, multioutput="raw_values")
        for i, label_name in enumerate(label_columns):
            print(f"Mean Squared Error (Normalized) - Multi Sequence Prediction Feedback Model for {label_name}: {MSE_normalized[i]}")
            model_results["multi_sequence_feedback"].update({"MSE_Normalized": [label_name, MSE_normalized[i]]})
        MSE_de_normalized = mean_squared_error(y_multi_seq_de_norm, predict_multi_seq_de_norm, multioutput="raw_values")       
        for i, label_name in enumerate(label_columns):
            print(f"Mean Squared Error (De_Normalized) - Multi Sequence Prediction Feedback Model for {label_name}: {MSE_de_normalized[i]}")
            model_results["multi_sequence_feedback"].update({"MSE_De_Normalized": [label_name, MSE_de_normalized[i]]})

    # # Save Model & Results
    # model_results_file = os.path.join("model", "model_output", "model_results.json")
    # with open(model_results_file, 'w') as results:
    #     results.write(json.dumps(model_results, cls=NumpyArrayEncoder))
    
    if save_model:
        model_save_dir = os.path.join(args.model_dir, "saved_models", time_stamp)
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)
        my_model.model.save(os.path.join(model_save_dir, "single_step"))
        mult_seq_oneshot_model.model.save(os.path.join(model_save_dir, "multi_step"))

    # Plot Loss
    time_stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = configs.data["ticker"] + "_"+ time_stamp + ".html"
    plot_path = os.path.join("plots", "learning_curves", filename)
    with open(plot_path, 'a') as f:
        if "single_point" in configs.training["training_type"]:
            f.write(single_step_loss_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        if "multi_sequence_oneshot" in configs.training["training_type"]:
            f.write(multi_seq_loss_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        if "multi_sequence_feedback" in configs.training["training_type"]:
            f.write(multi_seq_loss_feedback_fig.to_html(full_html=False, include_plotlyjs='cdn'))



    # Plotting Predictions
    if configs.plotting["save_plots"]:
        filename = configs.plotting["filename"] + "_" + time_stamp + ".html"
        plot_path = os.path.join("plots", "predictions", filename)
        with open(plot_path, 'a') as f:
            if "single_point" in configs.training["training_type"]:
                single_point_fig = plot_label_prediction(labels=de_normalize_labels, predictions=de_normalize_predictions, label_name="Close", configs=configs, model_name="Single Point")
                f.write(single_point_fig.to_html(full_html=False, include_plotlyjs='cdn'))
            if "multi_sequence_oneshot" in configs.training["training_type"]:
                multi_seq_one_shot_fig = plot_label_prediction(labels=de_normalize_labels_multi_seq_oneshot, predictions=de_normalize_predictions_multi_seq_oneshot, label_name="Close", configs=configs, model_name="Multi Sequence")
                f.write(multi_seq_one_shot_fig.to_html(full_html=False, include_plotlyjs='cdn'))
            if "multi_sequence_feedback" in configs.training["training_type"]:
                multi_seq_fig = plot_label_prediction(labels=y_multi_seq_de_norm, predictions=predict_multi_seq_de_norm, label_name="Close", configs=configs, model_name="Feedback Model")
                f.write(multi_seq_fig.to_html(full_html=False, include_plotlyjs='cdn'))


import plotly.graph_objects as go

def plot_label_prediction(labels, predictions, label_name, configs, model_name):
    """Plotting point by point prediction with labels
    
    Arguments:
        predictions: (np.array) point-by-point predictions
        labels: (np.array) true labels corresponding to y 
        label_name: (str) label name
        configs: (Params) Configurations
        n: (int) number of predictions and labels to show in the plots
    """
    label_index = configs.data_generator["label_columns"].index(label_name)
    predictions = predictions[:, label_index]
    labels = labels[:, label_index]

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    n = configs.plotting["n_points"]
    n = min(n, predictions.shape[0])

    fig = go.Figure()

    fig.add_trace(go.Scatter(y=labels[:n], name="True Data", mode="lines"))
    fig.add_trace(go.Scatter(y=predictions[:n] , name="Predictions", mode="lines",
                                line=dict(dash="dash")))

    fig.update_layout(title = model_name + "Predictions",
                        xaxis_title="Time Frame",
                        yaxis_title=label_name,
                        autosize=True)

    return fig



def plot_loss(train_loss, val_loss, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_loss, name="Train Loss", mode="lines"))
    fig.add_trace(go.Scatter(y=val_loss , name="Val Loss", mode="lines"))

    fig.update_layout(title = model_name, xaxis_title="Epochs", yaxis_title="Loss", autosize=True)
    return fig
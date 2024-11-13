# Should have RMSE, Demograpgic parity
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from fairlearn.metrics import MetricFrame, demographic_parity_difference



# Calculate RMSE
def calculate_rmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse

# Evaluate Demographic Parity (need predictions of gender for this case)
def evaluate_demographic_parity(y_true, y_pred, sensitive_features):
    # Example: Difference in mean predictions
    metric_frame = MetricFrame(
        metrics={"mean_prediction": np.mean},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    # Calculate the difference between groups
    parity_diff = metric_frame.difference(method='between_groups', metric="mean_prediction")
    
    return parity_diff
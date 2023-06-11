import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
def r_squared(y_true, y_pred):
    ssr = torch.sum((y_true - y_pred) ** 2)
    sst = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2_score = 1 - ssr / sst
    return r2_score
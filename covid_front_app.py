import pandas as pd
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve,precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score, balanced_accuracy_score



# Header and text
st.title("Diabetes prediction dataset DASHBOARD")
st.write("""This dashboard will present the info about the Diabetes prediction dataset from Kaggle (https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).""")

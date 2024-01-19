import pandas as pd
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve,precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score, balanced_accuracy_score
import plotly.graph_objects as go
from io import StringIO



# Header and text
st.title("Quick comparison of some algorithms using the Covid dataset as an example")
st.write("""This dashboard will present the info about the COVID-19 dataset, taken from 
Kaggle platform (https://www.kaggle.com/datasets/meirnizri/covid19-dataset)).""")

st.markdown("""
 * Use the menu at left to select data and set plot parameters
 * Your plots will appear below
""")


# -- Notes on whitening
with st.expander("See data description"):
    st.markdown("""
 * sex: 1 for female and 2 for male.
 * age: of the patient.
 * classification: covid test findings. Values 1-3 mean that the patient was diagnosed with covid in different
 * degrees. 4 or higher means that the patient is not a carrier of covid or that the test is inconclusive.
 * patient type: type of care the patient received in the unit. 1 for returned home and 2 for hospitalization.
 * pneumonia: whether the patient already have air sacs inflammation or not.
 * pregnancy: whether the patient is pregnant or not.
 * diabetes: whether the patient has diabetes or not.
 * copd: Indicates whether the patient has Chronic obstructive pulmonary disease or not.
 * asthma: whether the patient has asthma or not.
 * inmsupr: whether the patient is immunosuppressed or not.
 * hypertension: whether the patient has hypertension or not.
 * cardiovascular: whether the patient has heart or blood vessels related disease.
 * renal chronic: whether the patient has chronic renal disease or not.
 * other disease: whether the patient has other disease or not.
 * obesity: whether the patient is obese or not.
 * tobacco: whether the patient is a tobacco user.
 * usmr: Indicates whether the patient treated medical units of the first, second or third level.
 * medical unit: type of institution of the National Health System that provided the care.
 * intubed: whether the patient was connected to the ventilator.
 * icu: Indicates whether the patient had been admitted to an Intensive Care Unit.

""")

st.markdown("""
 We predict survival from covid. Target - 'date died': If the patient survived - 0, and 1 otherwise.
""")


# -- Notes on whitening
with st.expander("See notes"):
    st.markdown("""
 * Wtextount of noise. 
  [Signal Processing Tutorial](https://share.streamlit.io/jkanner/streamlit-audio/main/app.py)
""")

# sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    This app is Training dashboard.
    """
)



page = st.sidebar.selectbox("Choose page",
                            ["Charts",
                             "Make prediction"])

# New Line
def new_line(n=1):
    for i in range(n):
        st.write("\n")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload y_pred.csv file", type=["csv"])

# Placeholder for dataframes
y = pd.DataFrame()
y_scores = pd.DataFrame()

button_answer = st.sidebar.button('Show data')



if page == "Charts":
    st.header("""Charts Demo""")

    y = pd.read_csv('y.csv')
    y_pred_catb = pd.read_csv('y_pred_catb.csv')
    y_pred_knn = pd.read_csv('y_pred_knn.csv')
    y_pred_xgb = pd.read_csv('y_pred_xgb.csv')
    y_pred_lgbm = pd.read_csv('y_pred_lgbm.csv')
    y_pred_logreg = pd.read_csv('y_pred_logreg.csv')
    y_pred_nb = pd.read_csv('y_pred_nb.csv')

    # Чекбоксы для выбора моделей
    use_xgb_model = st.checkbox("Use XGBoost Model", value=False)
    use_lgbm_model = st.checkbox("Use LightGBM Model", value=False)
    use_catb_model = st.checkbox("Use CatBoost Model", value=False)
    use_logreg_model = st.checkbox("Use Logistic Regression Model", value=False)
    use_nb_model = st.checkbox("Use Naive Bayes Model", value=False)
    use_knn_model = st.checkbox("Use KNN Model", value=False)

    # Button to plot ROC AUC curves
    if st.button("Plot ROC AUC Curves"):

        # Plotting ROC AUC curves for selected models
        fig = go.Figure()

        if use_xgb_model:
            fpr, tpr, _ = roc_curve(y, y_pred_xgb)
            roc_auc = roc_auc_score(y, y_pred_xgb)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'XGBoost - ROC AUC: {roc_auc:.2f}'))

        if use_lgbm_model:
            fpr, tpr, _ = roc_curve(y, y_pred_lgbm)
            roc_auc = roc_auc_score(y, y_pred_lgbm)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'LightGBM - ROC AUC: {roc_auc:.2f}'))

        if use_catb_model:
            fpr, tpr, _ = roc_curve(y, y_pred_catb)
            roc_auc = roc_auc_score(y, y_pred_catb)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'CatBoost - ROC AUC: {roc_auc:.2f}'))

        if use_logreg_model:
            fpr, tpr, _ = roc_curve(y, y_pred_logreg)
            roc_auc = roc_auc_score(y, y_pred_logreg)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Logistic Regression - ROC AUC: {roc_auc:.2f}'))

        if use_nb_model:
            fpr, tpr, _ = roc_curve(y, y_pred_nb)
            roc_auc = roc_auc_score(y, y_pred_nb)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Naive Bayes - ROC AUC: {roc_auc:.2f}'))

        if use_knn_model:
            fpr, tpr, _ = roc_curve(y, y_pred_knn)
            roc_auc = roc_auc_score(y, y_pred_knn)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'KNN - ROC AUC: {roc_auc:.2f}'))

        # Updating layout
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curves',
            xaxis=dict(title='False Positive Rate'),
            yaxis=dict(title='True Positive Rate'),
            showlegend=True
        )

        # Displaying the plot in Streamlit
        st.plotly_chart(fig)

    # if not y_pred.empty:
    #     st.write('f1_score:', np.round(f1_score(y, y_pred), 3))
    #     st.write('r2_score:', np.round(r2_score(y_pred, y), 3))
    #     st.write('balanced_accuracy_score:', np.round(balanced_accuracy_score(y_pred, y), 3))
    #
    #     # Calculate ROC AUC
    #     fpr, tpr, thresholds = roc_curve(y, y_pred)
    #     roc_auc = roc_auc_score(y, y_pred)
    #
    #     # Plot ROC curve
    #     plt.figure(figsize=(8, 8))
    #     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic (ROC) Curve')
    #     plt.legend(loc='lower right')
    #     st.pyplot(plt)
    #
    #     # Plot precision recall
    #     # calculate precision and recall
    #     precision, recall, thresholds = precision_recall_curve(y_pred, y)
    #
    #     # create precision recall curve
    #     fig, ax = plt.subplots()
    #     ax.plot(recall, precision, color='purple')
    #
    #     # add axis labels to plot
    #     ax.set_title('Precision-Recall Curve')
    #     ax.set_ylabel('Precision')
    #     ax.set_xlabel('Recall')
    #
    #     # display plot
    #     st.pyplot(plt)
    #
    #     # Confusion Matrix
    #     cm = confusion_matrix(y, y_pred)
    #
    #     # Normalize the confusion matrix
    #     cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #
    #     # Display the confusion matrix
    #     fig, ax = plt.subplots()
    #     ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
    #     ax.set_title('Confusion Matrix (Normalized)')
    #     st.pyplot(fig)
    #
    # y_pred = pd.read_csv('y_pred.csv')
    # y = pd.read_csv('y.csv')
    #
    # st.write('f1_score:',np.round(f1_score(y, y_pred),3))
    # st.write('r2_score:',np.round(r2_score(y_pred, y),3))
    # st.write('balanced_accuracy_score:',np.round(balanced_accuracy_score(y_pred,y),3))
    #
    #
    # # Calculate ROC AUC
    # fpr, tpr, thresholds = roc_curve(y, y_pred)
    # roc_auc = roc_auc_score(y, y_pred)
    #
    # # Plot ROC curve
    # plt.figure(figsize=(8, 8))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    # st.pyplot(plt)
    #
    # # Plot precision recall
    # #calculate precision and recall
    # precision, recall, thresholds = precision_recall_curve(y_pred, y)
    #
    # #create precision recall curve
    # fig, ax = plt.subplots()
    # ax.plot(recall, precision, color='purple')
    #
    # #add axis labels to plot
    # ax.set_title('Precision-Recall Curve')
    # ax.set_ylabel('Precision')
    # ax.set_xlabel('Recall')
    #
    # #display plot
    # st.pyplot(plt)
    #
    #
    # # Confusion Matrix
    # cm = confusion_matrix(y, y_pred)
    #
    # # Normalize the confusion matrix
    # cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #
    # # Display the confusion matrix
    # fig, ax = plt.subplots()
    # ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
    # ax.set_title('Confusion Matrix (Normalized)')
    # st.pyplot(fig)






if page == "Make prediction":

    st.header("Select your model and make prediction:")

    # Ваши данные
    # y = pd.read_csv('y.csv')
    X = pd.read_csv(uploaded_file)
    X_dict = X.reset_index().to_dict(orient='list')

    if button_answer:
        st.write(X)

    # Чекбокс для выбора модели
    use_xgb_model = st.checkbox("Use XGBoost Model", value=False)
    use_lgbm_model = st.checkbox("Use LightGBM Model", value=False)
    use_catb_model = st.checkbox("Use CatBoost Model", value=False)
    use_logreg_model = st.checkbox("Use Logistic Regression Model", value=False)
    use_knn_model = st.checkbox("Use KNN Model", value=False)
    use_nb_model = st.checkbox("Use Naive Bayes Model", value=False)
    use_ens_model = st.checkbox("Use Ensemble Model", value=False)

    # Кнопка для отправки запроса
    if st.button("Make Prediction"):
        # Отправка запроса в зависимости от выбранной модели
        if use_xgb_model:
            endpoint = 'https://covidapp-bmvhwrbbeq-lm.a.run.app/model/predict_xgb'
        elif use_lgbm_model:
            endpoint = 'https://covidapp-bmvhwrbbeq-lm.a.run.app/model/predict_lgbm'
        elif use_catb_model:
            endpoint = 'https://covidapp-bmvhwrbbeq-lm.a.run.app/model/predict_catb'
        elif use_logreg_model:
            endpoint = 'https://covidapp-bmvhwrbbeq-lm.a.run.app/model/predict_logreg'
        elif use_knn_model:
            endpoint = 'https://covidapp-bmvhwrbbeq-lm.a.run.app/model/predict_knn'
        elif use_nb_model:
            endpoint = 'https://covidapp-bmvhwrbbeq-lm.a.run.app/model/predict_nb'
        elif use_ens_model:
            endpoint = 'https://covidapp-bmvhwrbbeq-lm.a.run.app/model/predict_ens'

        else:
            # Добавьте здесь другие модели, если необходимо
            endpoint = 'https://covidapp-bmvhwrbbeq-lm.a.run.app/model/predict_default'

        predicted_data_response = requests.post(endpoint, json=X_dict)

        # Проверка успешности запроса
        if predicted_data_response.status_code == 200:
            y_scores = pd.read_json(StringIO(predicted_data_response.json())).set_index('index')

            st.dataframe(y_scores)

            #
            # # Рассчет ROC AUC
            # fpr, tpr, thresholds = roc_curve(y, y_scores)
            # roc_auc = roc_auc_score(y, y_scores)
            #
            # # Создание графика Plotly
            # fig = go.Figure()
            # fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve'))
            # fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            # fig.update_layout(
            #     title='Receiver Operating Characteristic (ROC) Curve',
            #     xaxis=dict(title='False Positive Rate'),
            #     yaxis=dict(title='True Positive Rate'),
            #     showlegend=True
            # )
            #
            # # Отображение графика в Streamlit
            # st.plotly_chart(fig)
            # st.write(f'ROC AUC Score: {roc_auc:.2f}')
            #
            # # Confusion Matrix
            # cm = confusion_matrix(y, y_scores)
            #
            # # Normalize the confusion matrix
            # cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            #
            # # Display the confusion matrix
            # fig, ax = plt.subplots()
            # ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
            # ax.set_title('Confusion Matrix (Normalized)')
            # st.pyplot(fig)

        else:
            st.error(f"Error in prediction request. Status code: {predicted_data_response.status_code}")





st.sidebar.info("Feel free to contact me\n"'\n'
                "[My GitHub](https://github.com/PeterOstr)\n"'\n'
                "[My Linkedin](https://www.linkedin.com/in/ostrikpeter/)\n"'\n'
                "[Or just text me in Telegram](https://t.me/Politejohn)\n"
                ".")
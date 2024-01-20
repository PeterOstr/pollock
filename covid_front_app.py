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
 * Use the menu at left to select Charts page or to make predictions
 * Your plots will appear below
""")


# -- Notes on data_set
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



# sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    This app is Training dashboard.
    """
)



page = st.sidebar.selectbox("Choose page",
                            ["Info",
                             "Charts",
                             "Make prediction"])

# New Line
def new_line(n=1):
    for i in range(n):
        st.write("\n")

# # Function to plot Confusion Matrix
# def plot_confusion_matrix(cm, model_name):
#     cm_percentage = cm.astype('float') / cm
#     # Display the confusion matrix
#     fig, ax = plt.subplots()
#     ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
#     ax.set_title(f'Confusion Matrix - {model_name} (Normalized)')
#     st.pyplot(fig)


# Placeholder for dataframes
y = pd.DataFrame()
y_scores = pd.DataFrame()



if page == "Info":
    st.header("""Some info about models""")

    st.markdown("""#### KNN    """)
    st.markdown(""" The K-Nearest Neighbors (KNN) algorithm is a popular machine learning technique used for 
    classification and regression tasks. It relies on the idea that similar data points tend to have similar labels or 
    values.

During the training phase, the KNN algorithm stores the entire training dataset as a reference. When making predictions,
 it calculates the distance between the input data point and all the training examples, using a chosen distance metric
  such as Euclidean distance.

Next, the algorithm identifies the K nearest neighbors to the input data point based on their distances. 
In the case of classification, the algorithm assigns the most common class label among the K neighbors as the 
predicted label for the input data point. For regression, it calculates the average or weighted average
 of the target values of the K neighbors to predict the value for the input data point.
 
 [For more information see here](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
       """)

    st.markdown("""#### XGBoost    """)
    st.markdown(""" XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, 
    flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost 
    provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and 
    accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond
     billions of examples.
 
 [For more information see here](https://en.wikipedia.org/wiki/XGBoost)
       """)

    st.markdown("""#### Catboost    """)
    st.markdown(""" Catboost is a boosted decision tree machine learning algorithm developed by Yandex. It works in the
     same way as other gradient boosted algorithms such as XGBoost but provides support out of the box for categorical
      variables, has a higher level of accuracy without tuning parameters and also offers GPU support to speed up 
      training.
 
 [For more information see here](https://en.wikipedia.org/wiki/CatBoost)
       """)

    st.markdown("""#### Light GBM    """)
    st.markdown(""" Light GBM is a gradient boosting framework that uses tree based learning algorithm.
Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree 
leaf-wise while other algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the
 same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm.
 
 [For more information see here](https://en.wikipedia.org/wiki/LightGBM)
       """)

    st.markdown("""#### Logistic Regression    """)
    st.markdown(""" Logistic Regression is a statistical method used for binary and multi-class classification problems.
     Despite its name, it is a classification algorithm rather than a regression one. It predicts the probability of an 
     instance belonging to a particular class, and then makes a discrete prediction based on a threshold.
 
 [For more information see here](https://en.wikipedia.org/wiki/Logistic_regression)
       """)

    st.markdown("""#### Naive Bayes    """)
    st.markdown(""" Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem, which calculates
     the probability of a hypothesis (class) given the observed evidence (features). The "naive" assumption in Naive 
     Bayes is that all features are conditionally independent given the class. This simplifying assumption significantly
      reduces computational complexity, making it computationally efficient. Naive Bayes is computationally efficient
       due to the independence assumption, making it particularly useful for large datasets.
 
 [For more information see here](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
       """)

    st.header("""Some info about charts""")
    st.markdown("""#### ROC curve    """)
    st.markdown(""" A receiver operating characteristic curve, or ROC curve, is a graphical plot that illustrates the 
    performance of a binary classifier model (can be used for multi class classification as well) at varying threshold 
    values.

The ROC curve is the plot of the true positive rate (TPR) against the false positive rate (FPR) at each threshold 
setting.
 
 [For more information see here](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
       """)

    st.markdown("""#### Confusion Matrix    """)
    st.markdown(""" In the field of machine learning and specifically the problem of statistical classification, a 
    confusion matrix, also known as error matrix,[11] is a specific table layout that allows visualization of the 
    performance of an algorithm, typically a supervised learning one; in unsupervised learning it is usually called a
     matching matrix.

Each row of the matrix represents the instances in an actual class while each column represents the instances in a 
predicted class, or vice versa – both variants are found in the literature.[12] The name stems from the fact that it
 makes it easy to see whether the system is confusing two classes (i.e. commonly mislabeling one as another).
 
 [For more information see here](https://en.wikipedia.org/wiki/Confusion_matrix)
       """)

    st.header("""Some info about metrics""")
    st.markdown("""#### Accuracy and Precision    """)
    st.markdown(""" 
    Accuracy and precision are two measures of observational error. Accuracy is how close a given set of measurements 
    (observations or readings) are to their true value, while precision is how close the measurements are to each other.

In other words, precision is a description of random errors, a measure of statistical variability.
 Accuracy has two definitions:

More commonly, it is a description of only systematic errors, a measure of statistical bias of a given measure of 
central tendency; low accuracy causes a difference between a result and a true value; ISO calls this trueness.

Alternatively, the International Organization for Standardization (ISO) defines accuracy as describing a 
combination of both types of observational error (random and systematic), so high accuracy requires both high
 precision and high trueness.
In the first, more common definition of "accuracy" above, the concept is independent of "precision", so a particular 
set of data can be said to be accurate, precise, both, or neither.

In simpler terms, given a statistical sample or set of data points from repeated measurements of the same quantity,
 the sample or set can be said to be accurate if their average is close to the true value of the quantity being
  measured, while the set can be said to be precise if their standard deviation is relatively small.
  
   [For more information see here](https://en.wikipedia.org/wiki/Accuracy_and_precision)
       """)

    st.markdown("""#### Recall    """)
    st.markdown(""" 
    Recall, also known as the true positive rate (TPR), is the percentage of data samples that a machine learning model 
    correctly identifies as belonging to a class of interest—the “positive class”—out of the total samples for that
     class.
 
 [For more information see here](https://en.wikipedia.org/wiki/Precision_and_recall)
       """)

    st.markdown("""#### F1 Score    """)
    st.markdown(""" 
    In statistical analysis of binary classification and information retrieval systems, the F-score or F-measure is a 
    measure of predictive performance. It is calculated from the precision and recall of the test, where the precision 
    is the number of true positive results divided by the number of all samples predicted to be positive, including 
    those not identified correctly, and the recall is the number of true positive results divided by the number of all
     samples that should have been identified as positive. Precision is also known as positive predictive value, and
      recall is also known as sensitivity in diagnostic binary classification.

The F1 score is the harmonic mean of the precision and recall. It thus symmetrically represents both precision and
 recall in one metric. The more generic 
F_{beta } score applies additional weights, valuing one of precision or recall more than the other.

The highest possible value of an F-score is 1.0, indicating perfect precision and recall, and the lowest 
possible value is 0, if either precision or recall are zero.
  
   [For more information see here](https://en.wikipedia.org/wiki/F-score)
  
       """)

    st.markdown("""#### R2 Score    """)
    st.markdown(""" 
    R-Squared (R² or the coefficient of determination) is a statistical measure in a regression model that determines 
    the proportion of variance in the dependent variable that can be explained by the independent variable. In other 
    words, r-squared shows how well the data fit the regression model (the goodness of fit).
  
   [For more information see here](https://en.wikipedia.org/wiki/Coefficient_of_determination)
         """)

    st.markdown("""#### Balanced Accuracy    """)
    st.markdown(""" 
    Balanced accuracy is a metric that one can use when evaluating how good a binary classifier is. It is especially 
    useful when the classes are imbalanced, i.e. one of the two classes appears a lot more often than the other. 
    This happens often in many settings such as anomaly detection and the presence of a disease.
  
   [For more information see here](https://statisticaloddsandends.wordpress.com/2020/01/23/what-is-balanced-accuracy/)
       """)

if page == "Charts":
    st.header("""Charts Demo""")

    with st.expander("See notes"):
        st.write("""
        * Below you can see three buttons:
            - Display a curve - to plot a ROC curve
            - Build a Confusion matrix - to plot a Confusion matrix
            - Show a table of parameters - to show a table of measures of predictive performance
        
        """)

    # y = pd.read_csv('y.csv')
    # y_pred_catb = pd.read_csv('data_strmlt/y_pred_catb.csv')
    # y_pred_knn = pd.read_csv('data_strmlt/y_pred_knn.csv')
    # y_pred_xgb = pd.read_csv('data_strmlt/y_pred_xgb.csv')
    # y_pred_lgbm = pd.read_csv('data_strmlt/y_pred_lgbm.csv')
    # y_pred_logreg = pd.read_csv('data_strmlt/y_pred_logreg.csv')
    # y_pred_nb = pd.read_csv('data_strmlt/y_pred_nb.csv')

    # downloading from data for charts from github
    y = pd.read_csv('https://raw.githubusercontent.com/PeterOstr/pollock/main/data_strmlt/y.csv')
    y_pred_catb = pd.read_csv('https://raw.githubusercontent.com/PeterOstr/pollock/main/data_strmlt/y_pred_catb.csv')
    y_pred_knn = pd.read_csv('https://raw.githubusercontent.com/PeterOstr/pollock/main/data_strmlt/y_pred_knn.csv')
    y_pred_xgb = pd.read_csv('https://raw.githubusercontent.com/PeterOstr/pollock/main/data_strmlt/y_pred_xgb.csv')
    y_pred_lgbm = pd.read_csv('https://raw.githubusercontent.com/PeterOstr/pollock/main/data_strmlt/y_pred_lgbm.csv')
    y_pred_logreg = pd.read_csv('https://raw.githubusercontent.com/PeterOstr/pollock/main/data_strmlt/y_pred_logreg.csv')
    y_pred_nb = pd.read_csv('https://raw.githubusercontent.com/PeterOstr/pollock/main/data_strmlt/y_pred_nb.csv')


    st.markdown("""#### ROC Curve (ROC)    """)

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

    st.markdown("""#### Confusion Matrix   """)

    # Button to plot ROC AUC curves
    if st.button("Plot Confusion Matrix"):
        # Plotting Confusion Matrix for selected models
        if use_xgb_model:
            cm = confusion_matrix(y, y_pred_xgb)
                # Normalize the confusion matrix
            cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # Display the confusion matrix
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix XGBoost (Normalized)')
            st.pyplot(fig)

        if use_lgbm_model:
            cm = confusion_matrix(y, y_pred_lgbm)
            # plot_confusion_matrix(cm, "LightGBM")
            cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # Display the confusion matrix
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix LightGBM (Normalized)')
            st.pyplot(fig)

        if use_catb_model:
            cm = confusion_matrix(y, y_pred_catb)
            cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # Display the confusion matrix
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix CatBoost (Normalized)')
            st.pyplot(fig)

        if use_logreg_model:
            cm = confusion_matrix(y, y_pred_logreg)
            # Display the confusion matrix
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix Logistic Regression (Normalized)')
            st.pyplot(fig)

        if use_nb_model:
            cm = confusion_matrix(y, y_pred_nb)
            # Display the confusion matrix
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix Naive Bayes (Normalized)')
            st.pyplot(fig)

        if use_knn_model:
            cm = confusion_matrix(y, y_pred_knn)
            # Display the confusion matrix
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix KNN (Normalized)')
            st.pyplot(fig)

    if st.button("Show Metrics Table"):
        # Create a DataFrame to store the metrics
        metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Recall', 'F1 Score', 'R2 Score', 'Balanced Accuracy'])

        # Calculate and add metrics for selected models
        if use_xgb_model:
            accuracy = accuracy_score(y, y_pred_xgb)
            recall = recall_score(y, y_pred_xgb)
            f1 = f1_score(y, y_pred_xgb)
            r2 = r2_score(y, y_pred_xgb)
            balanced_accuracy = balanced_accuracy_score(y, y_pred_xgb)
            metrics_df = metrics_df._append({'Model': 'XGBoost', 'Accuracy': accuracy, 'Recall': recall, 'F1 Score': f1, 'R2 Score': r2, 'Balanced Accuracy': balanced_accuracy}, ignore_index=True)

        if use_lgbm_model:
            accuracy = accuracy_score(y, y_pred_lgbm)
            recall = recall_score(y, y_pred_lgbm)
            f1 = f1_score(y, y_pred_lgbm)
            r2 = r2_score(y, y_pred_lgbm)
            balanced_accuracy = balanced_accuracy_score(y, y_pred_lgbm)
            metrics_df = metrics_df._append({'Model': 'LightGBM', 'Accuracy': accuracy, 'Recall': recall, 'F1 Score': f1, 'R2 Score': r2, 'Balanced Accuracy': balanced_accuracy}, ignore_index=True)

        if use_catb_model:
            accuracy = accuracy_score(y, y_pred_catb)
            recall = recall_score(y, y_pred_catb)
            f1 = f1_score(y, y_pred_catb)
            r2 = r2_score(y, y_pred_catb)
            balanced_accuracy = balanced_accuracy_score(y, y_pred_catb)
            metrics_df = metrics_df._append({'Model': 'CatBoost', 'Accuracy': accuracy, 'Recall': recall, 'F1 Score': f1, 'R2 Score': r2, 'Balanced Accuracy': balanced_accuracy}, ignore_index=True)

        if use_logreg_model:
            accuracy = accuracy_score(y, y_pred_logreg)
            recall = recall_score(y, y_pred_logreg)
            f1 = f1_score(y, y_pred_logreg)
            r2 = r2_score(y, y_pred_logreg)
            balanced_accuracy = balanced_accuracy_score(y, y_pred_logreg)
            metrics_df = metrics_df._append({'Model': 'LogReg', 'Accuracy': accuracy, 'Recall': recall, 'F1 Score': f1, 'R2 Score': r2, 'Balanced Accuracy': balanced_accuracy}, ignore_index=True)

        if use_nb_model:
            accuracy = accuracy_score(y, y_pred_nb)
            recall = recall_score(y, y_pred_nb)
            f1 = f1_score(y, y_pred_nb)
            r2 = r2_score(y, y_pred_nb)
            balanced_accuracy = balanced_accuracy_score(y, y_pred_nb)
            metrics_df = metrics_df._append({'Model': 'CatBoost', 'Accuracy': accuracy, 'Recall': recall, 'F1 Score': f1, 'R2 Score': r2, 'Balanced Accuracy': balanced_accuracy}, ignore_index=True)

        if use_knn_model:
            accuracy = accuracy_score(y, y_pred_knn)
            recall = recall_score(y, y_pred_knn)
            f1 = f1_score(y, y_pred_knn)
            r2 = r2_score(y, y_pred_knn)
            balanced_accuracy = balanced_accuracy_score(y, y_pred_knn)
            metrics_df = metrics_df._append({'Model': 'KNN', 'Accuracy': accuracy, 'Recall': recall, 'F1 Score': f1, 'R2 Score': r2, 'Balanced Accuracy': balanced_accuracy}, ignore_index=True)

    # Repeat the above pattern for other models

        # Display the metrics table
        st.table(metrics_df)





if page == "Make prediction":

    uploaded_file = st.file_uploader("Choose a CSV file")

    st.header("Select your model and make prediction:")

    # # File uploader
    # uploaded_file = st.sidebar.file_uploader("Upload y_pred.csv file", type=["csv"])

    # Check if file is uploaded
    if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file is None:
        st.warning("Please upload a file before making predictions.")
        # uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        st.session_state.uploaded_file = uploaded_file

        button_answer = st.button('Show data')


    else:

        X = pd.read_csv(uploaded_file)
        X_dict = X.reset_index().to_dict(orient='list')
        st.session_state.uploaded_file = uploaded_file
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

            predicted_data_response = requests.post(endpoint, json=X_dict)
            results_predicted = pd.read_json(StringIO(predicted_data_response.json())).set_index('index')
            results_predicted.columns = ['Died']
            st.write('prediction results:')
            st.write(results_predicted)

#             # Проверка успешности запроса
#             if predicted_data_response.status_code == 200:
#                 y_scores = pd.read_json(StringIO(predicted_data_response.json())).set_index('index')
#
#                 st.dataframe(y_scores)
#
#                 #
#                 # # Рассчет ROC AUC
#                 # fpr, tpr, thresholds = roc_curve(y, y_scores)
#                 # roc_auc = roc_auc_score(y, y_scores)
#                 #
#                 # # Создание графика Plotly
#                 # fig = go.Figure()
#                 # fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve'))
#                 # fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
#                 # fig.update_layout(
#                 #     title='Receiver Operating Characteristic (ROC) Curve',
#                 #     xaxis=dict(title='False Positive Rate'),
#                 #     yaxis=dict(title='True Positive Rate'),
#                 #     showlegend=True
#                 # )
#                 #
#                 # # Отображение графика в Streamlit
#                 # st.plotly_chart(fig)
#                 # st.write(f'ROC AUC Score: {roc_auc:.2f}')
#                 #
#                 # # Confusion Matrix
#                 # cm = confusion_matrix(y, y_scores)
#                 #
#                 # # Normalize the confusion matrix
#                 # cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#                 #
#                 # # Display the confusion matrix
#                 # fig, ax = plt.subplots()
#                 # ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
#                 # ax.set_title('Confusion Matrix (Normalized)')
#                 # st.pyplot(fig)
#
#             else:
#                 st.error(f"Error in prediction request. Status code: {predicted_data_response.status_code}")



st.sidebar.info("Feel free to contact me\n"'\n'
                "[My GitHub](https://github.com/PeterOstr)\n"'\n'
                "[My Linkedin](https://www.linkedin.com/in/ostrikpeter/)\n"'\n'
                "[Or just text me in Telegram](https://t.me/Politejohn)\n"
                ".")
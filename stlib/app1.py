import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import xgboost as xgb
import vegafusion as vf

title = """1. Predict payment default risk"""
description = """**GOAL OF THIS PROJECT** : 

The goal of the project is to **analyze credit card customer data in Taiwan** and to **predict payment default risk** in the 
following month, using a **classification model** based on **XGBoost**. """

def run():
    # Header
    titolo = title
    st.markdown("<h1 style='text-align: left;'>"+str(titolo)+"</h1>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["**1. Description**","**2. Data Exploration**", "**3. Data Processing**", "**4. Modeling**", "**5. Result**"])
    
    with tab1:

        st.markdown("""
        The project is based on a dataset that contains:

        - Information about late payments
        - Demographic factors
        - Credit data
        - History of payments and billing statements of 30,000 credit card customers in Taiwan from April 2005 to September 2005.

        The dataset consists of 25 variables, including:

        - The `default.payment.next.month` target variable, which indicates whether or not the customer has paid the next month (1=yes, 0=no).
        - Other variables such as customer ID, credit limit, gender, education, marital status, age, and payment and billing details for the past six months.
                """)
        
        # Carica il dataframe dal file CSV
        st.markdown('### **Dataframe :**')
        df = pd.read_csv("stlib/files/UCI_Credit_Card.csv")
        st.dataframe(df)
        
    with tab2:
        # Utilizza st.expander() per creare un expander
        with st.expander("**Dataset Description**"):
            # Dimensioni del dataset
            st.markdown(" ##### Dataset shape : &emsp;  "+ str(df.shape))
            # Statistiche descrittive
            st.markdown(" ##### The descriptive statistics are : ")
            st.dataframe(df.describe())
            # Conteggio Unici mancanti
            st.markdown("##### Unique values per column are : ")
            st.dataframe(df.nunique().to_frame().T)
            # Conteggio valori mancanti
            st.markdown("##### Missing values per column : ")
            st.dataframe(df.isnull().sum().to_frame().T)
        with st.expander("**Data Plots**"):
            alt.data_transformers.disable_max_rows()
            # Imposta il limite di righe a 100000
            vf.enable(row_limit=100000)

            corr_df = df.corr(method='spearman').reset_index()
            columns = df.columns.tolist()
            melted_corr_df = corr_df.melt(id_vars=['index'], value_vars=columns, value_name="Value")       
            heatmap = alt.Chart(melted_corr_df).mark_rect().encode(
                x='index:O',
                y='variable:O',
                color=alt.Color('Value:Q'),
                tooltip="Value"
            ).properties(
                width=1000, height=700,
            ).interactive()

            #boxplot & histogram & heatmap
            col1, col2 = st.columns(2)
            with col1 :
                # creo uno selectbox con le colonne del df
                colonna = st.selectbox("Choose a Features", df.columns)
                # assumiamo che il dataframe df sia già definito
                boxplot = alt.Chart(df).mark_boxplot().encode(
                    x=colonna + ":Q",
                #x="LIMIT_BAL:Q",
                )

                histogram = alt.Chart(df).mark_bar().encode(
                alt.X(colonna + ":Q", bin=True),
                #alt.X("LIMIT_BAL:Q", bin=True),
                y='count()',
                )
                plots = (boxplot & histogram)
                st.altair_chart(plots, use_container_width=True)
            with col2 :
                st.altair_chart(heatmap, use_container_width=True)

    with tab3:
        st.markdown("""The following checks were carried out:
- Check for **missing data** and decide how to handle them. There are no missing values in this dataset.
- Check for **outliers** and decide how to handle them. We used the xgboost model and the 'roc_auc' metric to be able to handle the unbalanced dataset without transformations.
- The features_importances, available with xgboost, was done as a pre-process to be used later in the optimization of the model hyperparameters.
                    
The order of features by importance is : 
                    """)
        # Carica i dati dal file CSV
        data = pd.read_csv('stlib/files/importance.csv')
        
        # Crea un grafico a barre orizzontali con Altair
        bars = alt.Chart(data).mark_bar().encode(
            y=alt.Y('feature', title='Feature',sort='-x'),
            x=alt.X('importance:Q', title='Importance'),
            color=alt.Color('importance:Q', legend=None)
        )

        # Visualizza il grafico con Streamlit
        st.altair_chart(bars, use_container_width=True)

    with tab4:
        st.markdown("""To build and optimize the model, the XGBoost library was used, optimizing the hyper-parameters with Optuna, to create a **binary classification model**
                     which predicts the probability of the customer paying or not paying next month. To do this, you must follow these steps:
- Make a stratified cross-validation, with 5-fold, 3-repetition, to ensure that the distribution of the target variable is balanced across all sets.
- Create a **base model** using XGBoost default parameters, such as number of trees, maximum depth, learning rate, etc.
- Evaluate the **performance** of the base model using the metric appropriate for binary classification, ie the AUC, suitable for unbalanced features.
- Optimize the **parameters** of the model using automatic search techniques, in this case a Bayesian optimization, to find the combination
                     of parameters that maximizes the performance of the model.

The characteristics of the trained model are:
- The model has a **94%** probability of correctly classifying a random example between the two classes, using the first **17 features** in order of importance.
- The fact that the features have been reduced has led to a reduction of noise in the data and a generalization of it.
- The model occupies, in binary format '.xgb' only 2.7MB, very efficient from the point of view of performance in the case of deployment.
                    """)

    with tab5:
        # inizializza e carica il modello
        models = xgb.XGBClassifier() 
        models.load_model(r"stlib/files/model.xgb")
        st.markdown("#### Enter the data and check the result")
        labels = data.iloc[:-6,0].T
        values = [] #
        # creo inputbox affiancati
        col = st.columns(len(labels)) 
        # Usare il contesto with per inserire i widget in ogni colonna
        for i in range(len(labels)):
            with col[i]:
                values.append(st.text_input(labels[i]))
        # Verifica se i valori sono numeri e la lunghezza è corretta
        if all(values) == True: #
            print('ENTRATO IN IF')
            values = np.array(values).astype(float)
            predictions = models.predict(values.reshape(1, -1))
        else:
            predictions ='#'
        st.markdown(f"""<center> <b>The customer has paid the next month(1=yes, 0=no):</b>  <div style="font-size: 22px;  top: -5px;right:-20px; background-color: lightblue; width: 50px; border-radius: 0.4rem;">{predictions[0]}</div></center>""", unsafe_allow_html=True)
        st.markdown("\n")
        st.markdown(" #####  I show the dataset, from which to take the example data:")
        df_new = df.drop(columns='ID')
        df_new = df_new.reindex(columns=labels)
        df_new = df_new.iloc[:,:df.shape[1]- 6]
        st.dataframe(df_new,hide_index= True)

css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.5rem;
        margin-right: 30px;
        }
        div[data-testid="stExpander"] div[role="button"] p {
            font-size: 1.2rem;
            font-weight: 600 !important;
        }
    </style>
    '''

st.markdown(css, unsafe_allow_html=True)

if __name__ == "__main__":
    run()

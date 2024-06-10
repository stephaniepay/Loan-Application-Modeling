#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,make_scorer,classification_report
from sklearn.cluster import KMeans
from sklearn.naive_bayes import BernoulliNB

data = pd.read_csv("Bank_CreditScoring.csv")

def home():
    st.title('Loan Application Modeling') 
    st.header("Problem Statement")
    st.markdown('<div style="text-align: justify;">Loan is the main source of income for banks. The loan companies grant a loan after an intensive process of verification and validation. However, they still don’t have assurance if the applicant is able to repay the loan with no difficulties. Therefore, the main goal of this project is to build a predictive model to predict the eligibility of loan application for the bank customers based on their creditworthiness and personal details.</div>', unsafe_allow_html=True)
    st.text("")
    st.write("The data being used in this question is based on the **_Bank_CreditScoring.csv dataset_**.")
    st.text("")

# Reading csv file (Bank_CreditScoring.csv dataset)

# In[2]:
    st.header("Data")
    st.dataframe(data)
    st.text("")

# In[3]:
def part1():
    st.header("Overview of the dataset informtion")
    st.subheader("Data Shape")
    data.info()
    st.markdown("* There is a total number of 2350 rows and the 21 columns in the dataset.")
    st.markdown("* The variety of data type are int64 and object.")
    st.text("")

    st.subheader("Data Unique Values")
    st.write("The number of unique value in the column and the unique value for each column is shown below.\n\n")

    with st.expander("See more"):
        for column in data.columns:
            st.write("The number of unique value: ", len(data[column].unique()))
            st.write(column , ": " , data[column].unique(), "\n")
        
    st.text("")
    
def data_cleaning(): 
    # Pulau Pinang
    data["State"][data["State"] == "Penang"] = "Pulau Pinang"
    data["State"][data["State"] == "Pulau Penang"] = "Pulau Pinang"
    data["State"][data["State"] == "P.Pinang"] = "Pulau Pinang"

    # Kuala Lumpur
    data["State"][data["State"] == "K.L"] = "Kuala Lumpur"

    # N. Sembilan
    data["State"][data["State"] == "N.S"] = "N.Sembilan"

    # Sarawak
    data["State"][data["State"] == "SWK"] = "Sarawak"

    #Terengganu
    data["State"][data["State"] == "Trengganu"] = "Terengganu"

    #Johor Bahru
    data["State"][data["State"] == "Johor B"] = "Johor Bahru"
    data["State"][data["State"] == "Johor"] = "Johor Bahru"
    
    data['Employment_Type'] = pd.factorize(data['Employment_Type'])[0]
    data['More_Than_One_Products'] = pd.factorize(data['More_Than_One_Products'])[0]
    data['Property_Type'] = pd.factorize(data['Property_Type'])[0]
    data['State'] = pd.factorize(data['State'])[0]
    data['Decision'] = pd.factorize(data['Decision'])[0]
    

def part2(): 
    data = pd.read_csv("Bank_CreditScoring.csv")
    st.header("Data Preprocessing for Model Prediction")
    st.subheader("Data Cleaning")
    st.markdown("* Identify if there is any missing data in every column(incomplete data).")
    nullCheck = pd.DataFrame(data.isnull().sum(), columns=["Sum"])
    st.dataframe(nullCheck)
    st.text("")
    
    st.subheader("Data Transformation")
    # Pulau Pinang
    data["State"][data["State"] == "Penang"] = "Pulau Pinang"
    data["State"][data["State"] == "Pulau Penang"] = "Pulau Pinang"
    data["State"][data["State"] == "P.Pinang"] = "Pulau Pinang"

    # Kuala Lumpur
    data["State"][data["State"] == "K.L"] = "Kuala Lumpur"

    # N. Sembilan
    data["State"][data["State"] == "N.S"] = "N.Sembilan"

    # Sarawak
    data["State"][data["State"] == "SWK"] = "Sarawak"

    #Terengganu
    data["State"][data["State"] == "Trengganu"] = "Terengganu"

    #Johor Bahru
    data["State"][data["State"] == "Johor B"] = "Johor Bahru"
    data["State"][data["State"] == "Johor"] = "Johor Bahru"

    st.markdown("**Employment Type**  \n0:'employer', 1:'Self_Employed', 2:'government', 3:'employee', 4:'Fresh_Graduate'")

    st.markdown("**More_Than_One_Products**  \n0:'yes', 1:'no'")

    st.markdown("**Property_Type**  \n0:'condominium', 1:'bungalow', 2:'terrace', 3:'flat'")

    st.markdown("**State**  \n0:'Johor Bahru', 1:'Selangor', 2:'Kuala Lumpur', 3:'Pulau Pinang', 4:'N.Sembilan', 5:'Sarawak', 6:'Sabah', 7:'Terengganu', 8:'Kedah'")

    st.markdown("**Decision**  \n0:'Reject', 1:'Accept'")

    st.text("")

    data['Employment_Type'] = pd.factorize(data['Employment_Type'])[0]
    data['More_Than_One_Products'] = pd.factorize(data['More_Than_One_Products'])[0]
    data['Property_Type'] = pd.factorize(data['Property_Type'])[0]
    data['State'] = pd.factorize(data['State'])[0]
    data['Decision'] = pd.factorize(data['Decision'])[0]
    st.dataframe(data)
    st.text("")

def part3():
    data_cleaning()  
    st.header("Exploratory Data Analysis")
    st.dataframe(data.describe())
    st.text("")

def part4():
    data_cleaning()
    st.header("Data Visualization")
    st.subheader("Heatmap")
    with st.container():
        fig, ax = plt.subplots(figsize=(10,8))  
        sns.heatmap(data.corr(), cmap='Blues', linewidths=1, square=True, ax=ax)
        st.write(fig)

    st.subheader("Boxplot")
    st.write("The mean value for all numerical columns:")
    meanV = pd.DataFrame(data.mean(), columns=["Value"])
    st.dataframe(meanV)
    st.text("")
    with st.expander("Show boxplots"):
        for column in data.columns:
            if is_numeric_dtype(data[column]) == True:
                fig, ax = plt.subplots(figsize=(11,2))  
                sns.boxplot(data=data, x=column, ax=ax)
                st.write(fig)

    st.markdown("* For the column **'Years_for_Property_to_Completion'**, an upper outlier can be seen in the boxplot at the value of 13. However, the bar for value 13 in the histogram shows that there is a number of customer with 13 years for property to completion too. Therefore, it suppose not to be an outlier.")
    st.text("")

    st.subheader("Histogram")
    with st.expander("Show histograms"):
        for column in data.columns:
            if is_numeric_dtype(data[column]) == True:
                fig, ax = plt.subplots(figsize=(11,2))  
                sns.histplot(data=data, x=column, ax=ax)
                st.write(fig)
            
    data
        
    st.text("") 
    st.subheader("To plot a univariate distribution of observations and figure out the density of some important columns.")
    with st.expander("Show distplots"):
        st.write("1. Employment Type")
        fig, ax = plt.subplots(figsize=(10,4))  
        sns.distplot(data["Employment_Type"], ax=ax)
        st.write(fig)
        st.text("")
    
        st.write("2. Years to Financial Freedom")
        fig, ax = plt.subplots(figsize=(10,4))  
        sns.distplot(data["Years_to_Financial_Freedom"], ax=ax)
        st.write(fig)
        st.text("")
    
        st.write("3. Number of Properties")
        fig, ax = plt.subplots(figsize=(10,4))  
        sns.distplot(data["Number_of_Properties"], ax=ax)
        st.write(fig)
        st.text("")

        st.write("4. Property Type")
        fig, ax = plt.subplots(figsize=(10,4))  
        sns.distplot(data["Property_Type"], ax=ax)
        st.write(fig)
        st.text("")

        st.write("5. Years for Property to Completion")
        fig, ax = plt.subplots(figsize=(10,4))  
        sns.distplot(data["Years_for_Property_to_Completion"], ax=ax)
        st.write(fig)
        st.text("")

        st.write("6. Monthly Salary")
        fig, ax = plt.subplots(figsize=(10,4))  
        sns.distplot(data["Monthly_Salary"], ax=ax)
        st.write(fig)
        st.text("")

        st.write("7. Total Sum of Loan")
        fig, ax = plt.subplots(figsize=(10,4))  
        sns.distplot(data["Total_Sum_of_Loan"], ax=ax)
        st.write(fig)
        st.text("")

        st.write("8. Decision")
        fig, ax = plt.subplots(figsize=(10,4))  
        sns.distplot(data["Decision"], ax=ax)
        st.write(fig)
        st.text("")

    with st.expander("Show scatter plots"):
        st.write("Relationship between Decision and Years to Financial Freedom")
        fig, ax = plt.subplots(figsize=(10,5))  
        sns.scatterplot(x='Decision', y='Years_to_Financial_Freedom', data=data, ax=ax)
        st.write(fig)
        st.text("")

        st.write("Relationship between Monthly Salary and Loan Amount")
        fig, ax = plt.subplots(figsize=(10,5))  
        sns.scatterplot(x='Monthly_Salary',y='Loan_Amount', data=data, ax=ax) 
        st.write(fig)
        st.text("")
    
        st.write("Relationship between Employment Type and Total Sum of Loan")
        fig, ax = plt.subplots(figsize=(10,5))  
        sns.scatterplot(x='Employment_Type', y='Total_Sum_of_Loan', data=data, ax=ax)
        st.write(fig)
        st.text("")
    
        st.write("Relationship between Number of Loan to Approve and Score")
        fig, ax = plt.subplots(figsize=(10,5))  
        sns.scatterplot(x='Number_of_Loan_to_Approve', y='Score', data=data, ax=ax)
        st.write(fig)
        st.text("")
    
def part5():
    data_cleaning()    
    st.header("Data Prediction") 
    st.write("**Cluster Analysis Method: K-Means Clustering  \nClassification Method: Random Forest and Bernoulli Naive Bayes**")
    st.write("Following are the models developed:  \n1. K-Means Clustering and Random Forest  \n2. K-Means Clustering and Bernoulli Naive Bayes  \n")
    st.text("")
    st.subheader("K-Means Clustering and Random Forest Classifier")
    k_List = [2,3,4]
    accuracy_List = []

    x = data.drop("Decision", axis=1)
    y = data["Decision"]

    for i in range (3): 

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = 5)

        kmeans = (KMeans(n_clusters = k_List[i], random_state=5))
        kmeans.fit(x_train)
        y_labels_train = kmeans.labels_
        y_labels_test = kmeans.predict(x_test)
        x_train["KM_Clustering"] = y_labels_train
        x_test["KM_Clustering"] = y_labels_test

        randForestC = RandomForestClassifier(random_state = 5)

        randForestC.fit(x_train, y_train)
        ypred = randForestC.predict(x_test)
        accuracy = "{:.4f}".format(accuracy_score(y_test, ypred))
        st.write("Accuracy K=", i+1, ": ", accuracy)
        accuracy_List.append(accuracy)

    randomForestDF = pd.DataFrame(list(zip(k_List, accuracy_List)), columns=["K","Accuracy"])
    st.text("")
    st.markdown("<h1 style='font-weight:bold; font-size:20px;'>Output Visualization</h1>", unsafe_allow_html=True)
    st.dataframe(randomForestDF)
    st.text("")
    fig, ax = plt.subplots(figsize=(6,4))  
    sns.barplot(x = "Accuracy", y = "K", data=randomForestDF, ax=ax)
    st.write(fig)


    st.text("")
    st.subheader("K-Means Clustering and Bernoulli Naive Bayes Classifier")

    k_List = [2,3,4]
    accuracy_List = []

    x = data.drop("Decision", axis=1)
    y = data["Decision"]

    for i in range (3): 
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 5)

        kmeans = (KMeans(n_clusters = k_List[i],random_state=5))
        kmeans.fit(x_train)
        y_labels_train = kmeans.labels_
        y_labels_test = kmeans.predict(x_test)
        x_train['KM_Clustering'] = y_labels_train
        x_test['KM_Clustering'] = y_labels_test
    
        BerNB = BernoulliNB()

        BerNB.fit(x_train, y_train)
        ypred = BerNB.predict(x_test)
        accuracy = "{:.4f}".format(accuracy_score(y_test, ypred))
        st.write("Accuracy K =", i+1, ": ", accuracy)
        accuracy_List.append(accuracy)    
    
    BernouliNBDF = pd.DataFrame(list(zip(k_List, accuracy_List)), columns=["K","Accuracy"])
    st.text("")
    st.markdown("<h1 style='font-weight:bold; font-size:20px;'>Output Visualization</h1>", unsafe_allow_html=True)
    st.dataframe(BernouliNBDF)
    st.text("")
    fig, ax = plt.subplots(figsize=(6,4))  
    sns.barplot(x = "Accuracy", y = "K", data=BernouliNBDF, ax=ax)
    st.write(fig)
    st.text("")

    st.header("Accuracy Comparison and Discussion")
    st.markdown("<h1 style='font-size:16px;'>As we can see, the highest accuracy 76.31% is achieved by K-Means Clustering (K=3) and Bernoulli Naive Bayes Classifier.</h1>", unsafe_allow_html=True)


st.sidebar.markdown("<h1 style='text-align:center; font-size:18px'>TIC3151 ARTIFICIAL INTELLIGENCE  \nGROUP PROJECT</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align:center; font-size:16px'>Question 3</h1>", unsafe_allow_html=True)
option = st.sidebar.selectbox("Choose section:", ("Home", "Overview", "Data Preprocessing for Model Prediction", "Exploratory Data Analysis", "Data Visualization", "Data Prediction")) 

if option == "Home":
    home()
elif option == "Overview":
    part1()
elif option == "Data Preprocessing for Model Prediction":
    part2()
elif option == "Exploratory Data Analysis":
    part3()
elif option == "Data Visualization":
    part4()
elif option == "Data Prediction":
    part5() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


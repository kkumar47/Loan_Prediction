import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px




header = st.container()
rawdata = st.container()
eda = st.container()

with header:
	font="sans serif"
	textColor="#26273"
	st.title('Loan Default Prediction System')
  
def raw_data():
	return pd.read_csv('https://raw.githubusercontent.com/kkumar47/Loan_Prediction/master/lending_club_loan_two.csv')

rawdf = raw_data()

with rawdata:
	st.subheader("Lending Club Loan Dataset", anchor ='The Data')
	st.text('Raw Data')
	#The raw data is displayed here
	st.dataframe(rawdf.head(10))
	rawd = rawdf.to_csv().encode('utf-8')
	st.download_button('Download Data', data=rawd, file_name='Raw Data.csv', help='Download Data in CSV format')
	st.write(rawdf.corr())
	
	
with eda:
	st.subheader("Exploratory Data Analysis", anchor ='EDA')
	sns.set_theme(style="whitegrid")
	fig1 = plt.figure(figsize=(10,10))
	snsa = sns.countplot(x="loan_status", data=rawdf).set(title='Loan Data Distribution')
	plt.savefig('ouputa.png')
	st.pyplot(fig1)
	with open("ouputa.png", "rb") as file:
     			btn = st.download_button(
             		label="Download Plot",
             		data=file,
             		file_name="Data Distribution.png",
             		mime="image/png")
	fig2 = plt.figure(figsize=(10,10))
	snsb = sns.histplot(x="loan_amnt", data=rawdf, kde=True).set(title='Loan Amount Distribution')
	plt.savefig('ouputb.png')
	st.pyplot(fig2)
	with open("ouputb.png", "rb") as file:
     			btn = st.download_button(
             		label="Download Plot",
             		data=file,
             		file_name="Loan Amount Distribution.png",
             		mime="image/png")
	fig3 = plt.figure(figsize=(10,10))
	fig3 = px.pie(rawdf, values="loan_status", names="loan_status")
	st.pyplot(fig3)

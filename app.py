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
	st.subheader("Lending Club Loan Raw Dataset", anchor ='The Data')
	
	#The raw data is displayed here
	st.dataframe(rawdf.head(10))
	rawd = rawdf.to_csv().encode('utf-8')
	st.download_button('Download Data', data=rawd, file_name='Raw Data.csv', help='Download Data in CSV format')
	#st.write(rawdf.corr())
	
	
with eda:
	st.subheader("Exploratory Data Analysis", anchor ='EDA')
	sns.set_theme(style="whitegrid")
	ddist= st.selectbox('Select Data Distribution Plot', ('Overall Data','Loan Amount'), index =0, help='Select the distribution plot')
	
	if ddist == 'Overall Data':
		st.markdown('**_Loan Data Distribution_**')
		fig1 = plt.figure(figsize=(8,8))
		snsa = sns.countplot(x="loan_status", data=rawdf).set(title='Loan Data Distribution')
		plt.savefig('ouputa.png')
		st.pyplot(fig1)
		#fign = plt.figure(figsize=(5,5))
		#fign=px.bar(rawdf, x='loan_status')
		#st.plotly_chart(fign)
		with open("ouputa.png", "rb") as file:
     				btn = st.download_button(
             			label="Download Plot",
             			data=file,
             			file_name="Data Distribution.png",
             			mime="image/png")
	elif ddist == 'Loan Amount':
		st.markdown('**_Loan Amount Distribution_**')
		fig2 = plt.figure(figsize=(6,6))
		snsb = sns.histplot(x="loan_amnt", hue ='loan_status',data=rawdf, kde=True).set(title='Loan Amount Distribution')
		plt.savefig('ouputb.png')
		st.pyplot(fig2)
		with open("ouputb.png", "rb") as file:
     				btn = st.download_button(
             			label="Download Plot",
             			data=file,
             			file_name="Loan Amount Distribution.png",
             			mime="image/png")
	#fig3 = plt.figure(figsize=(10,10))
	#rawdf['loan_status'].value_counts().plot(kind='pie')
	#st.pyplot(fig3)
	st.markdown('**_Correlation Plot_**')
	fig3 = plt.figure(figsize=(10,10))
	snsc = sns.heatmap(rawdf.corr(), annot=True, cmap="viridis")
	plt.savefig('ouputc.png')
	st.pyplot(fig3)
	with open("ouputc.png", "rb") as file:
     			btn = st.download_button(
             		label="Download Plot",
             		data=file,
             		file_name="Correlation Plot.png",
             		mime="image/png")
	fig4 = plt.figure(figsize=(10,10))
	st.balloons()
	#plt.boxplot(rawdf['loan_amnt'])
	#st.pyplot(fig4)

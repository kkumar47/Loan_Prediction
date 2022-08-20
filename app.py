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
	
	col1, col2 = st.columns(2)
	with col1:
		st.markdown('**_Installment vs Loan Amount_**')
		fig4 = plt.figure(figsize=(6,6))
		snsd = sns.scatterplot(x="installment", y="loan_amnt", hue='loan_status', data=rawdf)
		plt.savefig('ouputd.png')
		st.pyplot(fig4)
		with open("ouputd.png", "rb") as file:
     				btn = st.download_button(
             			label="Download Plot",
             			data=file,
             			file_name="Installment vs Loan Amount.png",
             			mime="image/png")
	with col2:
		st.markdown('**_Loan Status vs Loan Amount_**')
		fig5 = plt.figure(figsize=(6,6))
		snse = sns.boxplot(x="loan_status", y="loan_amnt",  data=rawdf)
		plt.savefig('oupute.png')
		st.pyplot(fig5)
		with open("oupute.png", "rb") as file:
     				btn = st.download_button(
             			label="Download Plot",
             			data=file,
             			file_name="Box Plot.png",
             			mime="image/png")
	
	ggrade = st.radio('Select Model Optimizer',('Grade','Subgrade'))
	if ggrade == 'Grade':
		st.markdown('**_Loan Status vs Grade_**')
		fig6 = plt.figure(figsize=(8,8))
		snsf = sns.countplot(x="grade", data=rawdf, hue="loan_status")
		plt.savefig('ouputf.png')
		st.pyplot(fig6)
		with open("ouputf.png", "rb") as file:
     				btn = st.download_button(
             			label="Download Plot",
             			data=file,
             			file_name="Loan Status vs Grade.png",
             			mime="image/png")
	elif ggrade == 'Subgrade':
		st.markdown('**_Loan Status vs SubGrade_**')
		subgrade_order = sorted(rawdf["sub_grade"].unique())
		fig7 = plt.figure(figsize=(8,8))
		snsg = sns.countplot(x="sub_grade", data=rawdf, palette="coolwarm", order=subgrade_order, hue="loan_status")
		plt.savefig('ouputg.png')
		st.pyplot(fig7)
		with open("ouputg.png", "rb") as file:
     				btn = st.download_button(
             			label="Download Plot",
             			data=file,
             			file_name="Loan Status vs SubGrade.png",
             			mime="image/png")
	sugrade = st.radio('Do you want to Explore the Subgrades that does not get paid back a lot? ',('Yes','No'))
	if sugrade == 'Yes':
		st.markdown('_Exploring Grades that does not get paid back a lot_')
		fig8 = plt.figure(figsize=(8,8))
		f_g = rawdf[(rawdf["grade"]=="F") | (rawdf["grade"]=="G")]
		subgrade_order = sorted(f_g["sub_grade"].unique())
		snsg = sns.countplot(x="sub_grade", data=f_g, palette="coolwarm", order=subgrade_order, hue="loan_status")
		plt.savefig('ouputh.png')
		st.pyplot(fig8)
		with open("ouputh.png", "rb") as file:
     				btn = st.download_button(
             			label="Download Plot",
             			data=file,
             			file_name="Grade F&G.png",
             			mime="image/png")	   
	elif sugrade == 'No':
		st.markdown('_No Investigation Selected_')
	#st.balloons()
	#plt.boxplot(rawdf['loan_amnt'])
	#st.pyplot(fig4)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


header = st.container()
rawdata = st.container()
eda = st.container()
pprocess = st.container()
ttsplit = st.container()
smotet = st.container()
modelt = st.container()
modele = st.container()
modelg = st.container()

with header:
	font="sans serif"
	textColor="#26273"
	st.title('Loan Default Prediction System')
  
def raw_data():
	return pd.read_csv('https://raw.githubusercontent.com/kkumar47/Loan_Prediction/master/lending_club_loan_two.csv', nrows=200000)

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
	
	ggrade = st.radio('Select Grade Level',('Grade','Subgrade'))
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
		fig7 = plt.figure(figsize=(15,8))
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
		snsh = sns.countplot(x="sub_grade", data=f_g, palette="coolwarm", order=subgrade_order, hue="loan_status")
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
	
	suempl = st.radio('Do you want to check Employment length impact on loan repayment? ',('Yes','No'))
	if suempl == 'No':
		st.markdown('_Continuing without investigation_')
	elif suempl == 'Yes':
		col5, col6 = st.columns(2)
		with col5:
			st.markdown('**_Employment Length_**')
			fig10 = plt.figure(figsize=(8,8))
			emp_length_order = sorted(rawdf["emp_length"].dropna().unique())
			snsj = sns.countplot(x="emp_length",data=rawdf,order=emp_length_order)
			plt.savefig('ouputj.png')
			st.pyplot(fig10)
			with open("ouputj.png", "rb") as file:
     					btn = st.download_button(
             				label="Download Plot",
             				data=file,
             				file_name="Employment Length plot.png",
             				mime="image/png")
		with col6:
			st.markdown('**_Employment Length vs Loan Status_**')
			fig11 = plt.figure(figsize=(8,8))
			emp_length_order = sorted(rawdf["emp_length"].dropna().unique())
			snsk = sns.countplot(x="emp_length",data=rawdf,order=emp_length_order, hue="loan_status")
			plt.savefig('ouputk.png')
			st.pyplot(fig11)
			with open("ouputk.png", "rb") as file:
     					btn = st.download_button(
             				label="Download Plot",
             				data=file,
             				file_name="Employment Length vs Loan status plot.png",
             				mime="image/png")
				
	le = preprocessing.LabelEncoder()
	le.fit(rawdf['loan_status'])
	rawdf['loan_repaid']=le.transform(rawdf['loan_status'])
	st.write('Loan Status encoded')
	df = rawdf.drop_duplicates(['loan_repaid','loan_status'])[['loan_repaid','loan_status']]
	st.dataframe(df)
	rawdf = rawdf.drop("loan_status", axis=1)
	st.markdown('**_Correlation Plot with Encoded Loan Status-Loan Repaid_**')
	fig9 = plt.figure(figsize=(8,8))
	snsi = rawdf.corr()["loan_repaid"].sort_values().drop("loan_repaid").plot(kind="bar")
	plt.savefig('ouputi.png')
	st.pyplot(fig9)
	with open("ouputi.png", "rb") as file:
     			btn = st.download_button(
             		label="Download Plot",
             		data=file,
             		file_name="Encoded Correlation plot.png",
             		mime="image/png")

			
with pprocess:
	st.subheader("Pre-Process Data", anchor ='EDA')
	col3, col4 = st.columns(2)
	with col3:
		st.markdown('_Missing Value Count by Attribute_')
		st.write(rawdf.isnull().sum())
	with col4:
		st.markdown('_Missing Value Count by Attribute in percentage_')
		st.write(100* rawdf.isnull().sum()/len(rawdf))

	with st.spinner('Handling Nulls...'):
		#Dropping unwanted Columns
		rawdf = rawdf.drop("emp_title", axis=1)
		rawdf = rawdf.drop("emp_length", axis=1)
		rawdf = rawdf.drop("title", axis=1)
			#Handling Null
		total_acc_avg = rawdf.groupby("total_acc").mean()["mort_acc"]
		def fill_mort_acc(total_acc,mort_acc):
			if np.isnan(mort_acc):
				return total_acc_avg[total_acc]
			else:
				return mort_acc
		rawdf["mort_acc"] = rawdf.apply(lambda x: fill_mort_acc(x["total_acc"],x["mort_acc"]),axis=1)
		rawdf = rawdf.dropna()
	st.success('Null values Handled!!', icon="✅")
	st.markdown('_Missing Value count post Handling Null_')
	st.write(rawdf.isnull().sum())


	with st.spinner('Preprocessing Data...'):
		rawdf["term"] = rawdf["term"].apply(lambda term: 36 if term=='36 months' else 60)
		rawdf = rawdf.drop("grade", axis=1)
		dummies = pd.get_dummies(rawdf["sub_grade"],drop_first=True)
		rawdf = pd.concat([rawdf.drop("sub_grade", axis=1),dummies],axis=1)
		dummies = pd.get_dummies(rawdf[['verification_status', 'application_type','initial_list_status','purpose']],drop_first=True)
		rawdf = pd.concat([rawdf.drop(['verification_status', 'application_type','initial_list_status','purpose'], axis=1),dummies],axis=1)
		rawdf["home_ownership"] = rawdf["home_ownership"].replace(["NONE","ANY"],"OTHER")
		dummies = pd.get_dummies(rawdf["home_ownership"],drop_first=True)
		rawdf = pd.concat([rawdf.drop("home_ownership", axis=1),dummies],axis=1)
			
		rawdf["zipcode"] = rawdf["address"].apply(lambda adress : adress[-5:])
		dummies = pd.get_dummies(rawdf["zipcode"],drop_first=True)
		rawdf = pd.concat([rawdf.drop("zipcode", axis=1),dummies],axis=1)
		rawdf = rawdf.drop("address", axis=1)
		rawdf = rawdf.drop("issue_d", axis=1)
			
		rawdf["earliest_cr_line"] = rawdf["earliest_cr_line"].apply(lambda date: int(date[-4:]))

	st.success('Preprocess Successfull!!', icon="✅")
	st.dataframe(rawdf.head(10))



	
with ttsplit:
	st.subheader('Train-Test Split')
	X = rawdf.drop("loan_repaid", axis=1).values
	y = rawdf["loan_repaid"].values
	col7, col8 = st.columns(2)
	test = col7.slider('Select testing data ', min_value=0.1, max_value=0.3, value=0.1, step=0.05, help='Select the test data percentage by sliding the slider ')
	train= col8.metric(label ='Train data: ', value=1-test)	
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test, random_state=42)
	col7.write('Shape of Training data post data split')
	col7.write(X_train.shape)
	col8.write('Shape of Training Label post split')	
	col8.write(y_train.shape)
	#st.markdown('_Applying minority oversampling on Training data_')
	#with st.spinner('Applying Minority Oversampling with SMOTE...'):
		#oversample = SMOTE(random_state = 101)
		#X_train_ad, y_train_ad = oversample.fit_resample(X_train, y_train)
	#st.success('SMOTE Successfull!!', icon="✅")
	#col9,col10 = st.columns(2)
	#col9.write('Shape of Training data post SMOTE')
	#col9.write((X_train_ad.shape))
	#col10.write('Shape of Training label post SMOTE')
	#col10.write((y_train_ad.shape))
	#unique, counts = np.unique(y_train_ad, return_counts=True)
	#result = np.column_stack((unique, counts)) 
	#st.write((result))
		
with modelt:
	st.subheader('CNN Model Training')
	scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	

	with st.spinner('Training CNN Model...'):
		model = Sequential()
		model.add(Dense(78, activation="relu"))
		model.add(Dropout(0.2)) #preventing overfitting

		model.add(Dense(39, activation="relu")) #reducing number of neurons of a half
		model.add(Dropout(0.2))

		model.add(Dense(19, activation="relu"))
		model.add(Dropout(0.2))

		model.add(Dense(units=1, activation="sigmoid")) #because it's a binary classification

		model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
		model.fit(x = X_train, y = y_train, epochs = 25, batch_size = 256, validation_data=(X_test, y_test))
	st.success('Model Training Completed', icon="✅")
	losses = pd.DataFrame(model.history.history)
	st.dataframe(losses)
	col9, col10 = st.columns(2)
	with col9:
		fig12 = plt.figure(figsize=(8,8))
		plt.plot(losses['accuracy'], label='accuracy')
		plt.plot(losses['val_accuracy'], label='val_accuracy')
	
		plt.xlabel("Epoch #", fontsize = 20)
		plt.ylabel("Losses", fontsize = 20)
		plt.legend()
		plt.savefig('ouputl.png')
		st.pyplot(fig12)
		with open("ouputl.png", "rb") as file:
     				btn = st.download_button(
             			label="Download Plot",
             			data=file,
             			file_name="Model Accuracy.png",
             			mime="image/png")
	with col10:
		losses_l = losses[['loss','val_loss']]
		
		st.line_chart(losses_l)
		
with modele:
	st.subheader('CNN Model Evaluation')
	predictions = (model.predict(X_test) > 0.5).astype("int32")
	st.markdown('**_Classification Report_**')
	st.write(classification_report(y_test, predictions))
	st.markdown('**_Confusion Matrix_**')
	result =confusion_matrix(y_test,predictions)
	fig13,ax = plt.subplots(figsize=(15,8))
	snsj = sns.heatmap(result, annot=True, ax=ax)
	ax.set_xlabel('Predicted Class')
	ax.set_ylabel('Actual Class')
	ax.set_title('Confusion Matrix')
	plt.savefig("ouputj.png")
	st.pyplot(fig13)
	with open("ouputj.png", "rb") as file:
     				btn = st.download_button(
             			label="Download Plot",
             			data=file,
             			file_name="Confusion Matrix.png",
             			mime="image/png")
	
with modelg:
	st.subheader('Gradient Boost Model Training')
	with st.spinner('Training Gradient Boost Model...'):
		def evaluate_model(model, X, y):
			# define the evaluation procedure
			cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
			# evaluate the model and collect the results
			scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
			return scores
		def get_models():
			models = dict()
			# define number of trees to consider
			n_trees = [10, 50, 100, 500, 1000, 5000]
			for n in n_trees:
				models[str(n)] = GradientBoostingClassifier(n_estimators=n)
			return models
		models = get_models()
		results, names = list(), list()
		for name, model in models.items():
			# evaluate the model
			scores = evaluate_model(model, X_train, y_train)
			# store the results
			results.append(scores)
			names.append(name)
		#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
	st.success('Model Training Completed', icon="✅")
	st.markdown('_Mean Accuracy of Gradient Boost Model_')
	fig14 = plt.figure(figsize=(10,10))
	plt.boxplot(results, labels=names, showmeans=True)
	plt.xlabel('No. of Estimators')
	plt.ylabel('Accuracy')
	plt.savefig("ouputk.png")
	st.pyplot(fig14)
	with open("ouputk.png", "rb") as file:
     				btn = st.download_button(
             			label="Download Plot",
             			data=file,
             			file_name="Gradient Boost Accuracy Plot.png",
             			mime="image/png")
		    
	#score_gb = clf.score(X_test, y_test)
	#st.metric(label = 'Mean Accuracy',value=score_gb)

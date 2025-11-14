# import necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import pickle


#load data
data = pd.read_csv("data/True&FalseNews.csv")
print(data.head())

#Separate features and label
x = data['text']
y = data['class']

#Split data into training and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state =4)

#Conver text into TF-IVF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_tfvif = vectorizer.fit_transform(x_train)
x_test_tfvif = vectorizer.transform(x_test)


#Initialize and train model
mnb = MultinomialNB()
mnb.fit(x_train_tfvif, y_train)
mnb.score(x_train_tfvif, y_train)


#Predict
y_pred = mnb.predict(x_test_tfvif)
print(y_pred)


#model evaluation
a =  accuracy_score(y_pred, y_test)
print(f"Model Accuracy = {a}")
confusion_matrix(y_pred, y_test)
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=['fake', 'true']).plot()
import joblib


#Save the trained model and vectorizer
joblib.dump(mnb, filename="model/model.pkl")
joblib.dump(vectorizer, 'vectorizer.pkl')














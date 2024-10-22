import pandas as pd 
import numpy as np 
import streamlit as st
from sklearn.preprocessing import RobustScaler 
from sklearn.linear_model import LogisticRegression
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 30
        else:
            return 25
    else:
        return Age
age=st.sidebar.number_input("Person Age",min_value=0,max_value=80,value=18,step=1)
gender=st.sidebar.selectbox("Select gender of the person",("Male","Female"))
Sibsp=st.sidebar.number_input("Sibsp number",min_value=0,max_value=8,value=0,step=1)
Parch=st.sidebar.number_input("Number of parents/Children Aboard",min_value=0,max_value=6,value=0,step=1)
Embarked=st.sidebar.selectbox("Select port of embarkation",("Cherbourg","Queenstown","Southampton"))
Pclass=st.sidebar.selectbox("Select Passenger Class",(1,2,3))
Fare=st.sidebar.number_input("Input Passenger Fare",min_value=0,max_value=513,value=0,step=10)

#st.button("Modeli yenileyin", type="primary")
if st.button("Rerun"):
    st.experimental_rerun()
st.write("This page is rerun")
if gender=="Male":
    n_gender=1
else:
    n_gender=0
if Embarked=="Cherbourg":
    Q,S=0,0
elif Embarked=="Queenstown":
    Q,S=1,0
else:
    Q,S=0,1
data=dict(
    Pclass=Pclass,
    Age=age,
    SibSp=Sibsp,
    Parch=Parch,
    Fare=Fare,
    male=n_gender,
    Q=Q,
    S=S)
data_vis=dict(
    Pclass=Pclass,
    Age=age,
    SibSp=Sibsp,
    Parents_children=Parch,
    Fare=Fare,
    Gender=gender,
    Port=Embarked)
# data = {
#     'pclass': pclass,
#     'age': age,
#     'sibsp': sibsp,
#     'parch': parch,
#     'fare': fare,
#     'male': n_gender,
#     'q': q,
#     's': s
# }
# data_vis = {
#     'Pclass': pclass,
#     'Age': age,
#     'Sibsp': sibsp,
#     'Parents_children': parch,
#     'Fare': fare,
#     'Gender': gender,
#     'Port': embarked
# }



st.title("This app predicts whether person colud have survived on Titanic or not")
dv=pd.DataFrame(data_vis,index=["Information about person"])
df=pd.DataFrame(data,index=["Information about person"]).astype("float64")
st.dataframe(dv)
train=pd.read_csv("train.csv")
train["Age"]=train[["Age","Pclass"]].apply(impute_age,axis=1)
train.drop(["Cabin","PassengerId"],axis=1,inplace=True)
train.dropna(inplace=True)
sex=pd.get_dummies(train["Sex"],drop_first=True)
embark=pd.get_dummies(train["Embarked"],drop_first=True)
train.drop(["Sex","Embarked","Name","Ticket"],axis=1,inplace=True)
train=pd.concat([train,sex,embark],axis=1)
X=train.drop('Survived',axis=1).copy()
y=train["Survived"]
scaler=RobustScaler()
X.iloc[:,:]=scaler.fit_transform(X)
df.iloc[:,:]=scaler.transform(df)
logmodel=LogisticRegression(max_iter=100000)
logmodel.fit(X,y)
prediction=logmodel.predict(df)
predictions_probablities=logmodel.predict_proba(df)
max_prob=np.max(predictions_probablities)*100
predict=lambda a: "survived" if a==1 else 'did not survive'
st.markdown(f"The person presumably **{predict(prediction)}** with probablity of **{round(max_prob,2)}**%")
st.markdown("Created by Aydin Khudiyev")
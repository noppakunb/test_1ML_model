# import module
import streamlit as st
import numpy as np
import pickle

# Title
st.title("Iris classification")

sepal_height = st.text_input("Enter Sepal height :", "Type Here ...")
sepal_width = st.text_input("Enter sepal width :", "Type Here ...")
petal_height = st.text_input("Enter Petal height :", "Type Here ...")
petal_width = st.text_input("Enter Petal width :", "Type Here ...")
features = [sepal_height,sepal_width,petal_height,petal_width]
results = np.array(features).reshape(1, -1)

picklefile = open("decision_tree_iris.pkl", "rb")
model = pickle.load(picklefile)
name=""

if(st.button("Prediction")):
  iris_class = model.predict(results)
  if (iris_class==0):
    name="setosa"
  elif(iris_class==1):
    name ="versicolor"
  else:
    name="virginica"
st.success("Result is " + name)

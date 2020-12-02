import pandas as pd
import numpy as np
from array import *
#For inporting dataset to our program.
df = pd.read_csv('Data_Set.csv')
print ("***ORIGINAL DATA***")
print(df)

print(" ")
print(" ")
print(" ")
print(" ")

#For converting the data into numeric form, so that the program can understand the data and execute the required needs.
from sklearn.preprocessing import LabelEncoder
LabelEncoder_df = LabelEncoder()
df = df.apply(LabelEncoder().fit_transform)
print("*************AFTER CONVERTING IT INTO REQUIRED FORM*************")
print (df)

#Dropping the target column
Input = df.drop(['ID','Dry?'] , axis = 1).values 
Target = df.iloc[:,1].values

print(Input)
print(Target)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(Input,Target)

#Giving Input values!!!
Input_array = array('i',[])
for i in range(4):
	if i == 0:
		print("Enter precip1 value")
	if i == 1:
		print("Enter Potevap value")
	if i == 2:
		print("Enter Precip_2 value")
	if i == 3:
		print("Enter Field Readiness Assess value")
	x = int(input())
	Input_array.append(x)

Input_array = [Input_array]
Y_pred = classifier.predict(Input_array)
print(" ")
print(" ")
print("General KNN-class:")

if Y_pred == 0:
	print("DRY")
else:
	print("WET")

classifier = KNeighborsClassifier(n_neighbors = 3,weights = 'distance')
classifier.fit(Input,Target)
Y_pred = classifier.predict(Input_array)
print("distance KNN-class:")
if Y_pred == 0:
	print("DRY")
else:
	print("WET")


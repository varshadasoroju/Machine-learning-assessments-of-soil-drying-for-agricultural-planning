import pandas as pd
import numpy as np
from array import *
#For inporting dataset to our program.
df = pd.read_csv("Data_Set.csv", encoding = 'ANSI')
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
Input = df.drop(['ID','Dry?'], axis = 1)    
Target = df.iloc[:,1]
print(Input)
print(Target)

#CONDITIONS that represent WET.
#Precip_2>=5;Potevap<3;7<=precip1<=9; is the order of decision tree.

from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier()
regressor.fit(Input,Target)

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

Y_pred = regressor.predict([Input_array])
print(" ")
print(" ")
if Y_pred == 0:
	print("DRY")
else:
	print("WET")


#For generating DECISIONTREE!!
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(regressor, out_file = dot_data, filled = True, rounded = True, special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
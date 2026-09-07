import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv("student_performance.csv")
print("first 5 rows")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\ncolumn name")
print(df.columns)
print("\nDataset Information:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
df["Attendance"]=df["Attendance"].fillna(df["Attendance"].mean())
df["Assignments"]=df["Assignments"].fillna(df["Assignments"].mean())
df["PreviousMarks"]=df["PreviousMarks"].fillna(df["PreviousMarks"].mean())
print(df["Assignments"])
print(df["Attendance"])
print(df["PreviousMarks"])
print("\nDuplicate Rows:")
df.duplicated().sum()
df=df.drop_duplicates()

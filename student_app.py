import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
# Load dataset
df = pd.read_csv(r"C:\Users\comp\Desktop\Student_performance_data _ (2).csv")

# Display first few rows
df.head()

# Define decoding mappings (example mappings)
grade_class_map = {0: 'A', 1: 'B', 2: 'C'}
#fee_status_map = {0: 'Unpaid', 1: 'Partial', 2: 'Paid'}
parental_support_map = {0: 'Low', 1: 'Medium', 2: 'High'}
tutoring_map = {0: 'No', 1: 'Yes'}

# Convert numerical values back to categorical
df["GradeClass"] = df["GradeClass"].map(grade_class_map)
#df["Fee Payment Status"] = df["ParentalSupport"].map(fee_status_map)
df["ParentalSupport"] = df["ParentalSupport"].map(parental_support_map)
df["Tutoring"] = df["Tutoring"].map(tutoring_map)

# Display modified dataset
df.head()
# Define possible fee status categories
fee_status_categories = ["Unpaid", "Partial", "Paid"]

# Assign random fee statuses to each student
df["FeePaymentStatus"] = np.random.choice(fee_status_categories, size=len(df))

# Display updated dataset
df.head()
# Label Encoding for Categorical Data
categorical_cols = ["GradeClass", "FeePaymentStatus","ParentalSupport", "Tutoring"]
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Display dataset after encoding
df.head()
#Using Sampling (Displaying a Subset of Students)
#Instead of plotting all students,show only every 10th student:
# Sample every 10th student to reduce clutter

df_sampled = df.iloc[::10]  

plt.figure(figsize=(8, 5))
plt.plot(df_sampled["StudentID"], df_sampled["Absences"], marker='o', linestyle='-', color='b')

plt.xlabel("Student ID (Sampled)")
plt.ylabel("Absences")
plt.title("Student Absenteeism Trends (Sampled)")
plt.grid(True)
plt.show()

#If some students still have high absences, they might need academic support.
#Apply rolling mean (window = 10 students)
#A rolling mean (moving average) smooths out the fluctuations, making trends easier to see.
#If a specific group has higher absences, focus on their extracurricular workload or parental support

df_sorted = df.sort_values(by="StudentID")  # Ensure data is sorted

df_sorted["Absences_Smoothed"] = df_sorted["Absences"].rolling(window=10, min_periods=1).mean()

plt.figure(figsize=(8, 5))
plt.plot(df_sorted["StudentID"], df_sorted["Absences_Smoothed"], marker='o', linestyle='-', color='r')

plt.xlabel("Student ID")
plt.ylabel("Smoothed Absences")
plt.title("Smoothed Student Absenteeism Trends")
plt.grid(True)
plt.show()
#Aggregating Absences by Grade Class (Better Grouping)
#Instead of plotting every student, group them by Grade Class and take the average absences per grade.
# Group data by Grade Class and take mean absences

df_grouped = df.groupby("GradeClass")["Absences"].mean()

plt.figure(figsize=(8, 5))
df_grouped.plot(kind="bar", color="purple", edgecolor="black")

plt.xlabel("Grade Class")
plt.ylabel("Average Absences")
plt.title("Average Absences by Grade Class")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()
# Histogram showing the GPA of students with respect to their absence count 

plt.figure(figsize=(8,5))
sns.histplot(df["GPA"], bins=20, kde=True, color="blue")
plt.title("GPA Distribution")
plt.show()

# Shwoing the fluctations in results(GPA) according to the count of absencees of students. 
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["Absences"], y=df["GPA"], hue=df["GradeClass"])
plt.title("Absenteeism vs. GPA")
plt.show()
# Title of the Dashboard
st.title("📊 Student Performance Dashboard")

# Sidebar Filters
st.sidebar.header("Filter Data")
grade_filter = st.sidebar.selectbox("Select Grade Class", df["GradeClass"].unique())

# Filter dataset based on selection
filtered_df = df[df["GradeClass"] == grade_filter]

# Display student summary
st.subheader("🎯 Student Overview")
st.dataframe(filtered_df[["StudentID", "GPA", "FeePaymentStatus", "Absences", "Tutoring"]])

# Attendance vs GPA Plot
st.subheader("📌 Attendance vs GPA")
fig = px.scatter(df, x="Absences", y="GPA", color="GradeClass", title="Impact of Absenteeism on GPA")
st.plotly_chart(fig)

# Fee Payment Status
st.subheader("💰 FeePaymentStatus")
fee_status_counts = df["FeePaymentStatus"].value_counts()
st.bar_chart(fee_status_counts)

# Alerts Section
st.subheader("🚨 Important Alerts")
if df["GPA"].mean() < 2.0:
    st.error("⚠️ Warning: Average GPA is low. Intervention needed!")
else:
    st.success("✅ Overall GPA is stable.")

# Extracurricular Activities
st.subheader("🎭 Extracurricular Participation")
activity_counts = df[["Sports", "Music", "Volunteering"]].sum()
st.bar_chart(activity_counts)

# Footer
st.markdown("Developed by [Your Name] 🚀")

import joblib
import pandas as pd
import customtkinter as ctk
from tkinter import ttk

# Load pre-trained model, scaler, and feature columns
model = joblib.load('Loan_decision_tree.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Extract occupation list from feature columns
all_occupations = [col.replace('occupation_', '') for col in feature_columns if col.startswith('occupation_')]

# Load dataset for display
data = pd.read_csv('loan.csv').head(30)  # Load top 30 records for demonstration

# Remove credit_score from feature columns
feature_columns = [col for col in feature_columns if col != 'credit_score']


def predict_loan_status():
    try:
        # Collect input values from GUI
        age = age_entry.get()
        gender = gender_combobox.get()
        income = income_entry.get()
        education_level = education_combobox.get()
        marital_status = marital_combobox.get()
        occupation = occupation_combobox.get()

        # Validate inputs
        if not age.isdigit() or not income.replace('.', '', 1).isdigit():
            result_label.configure(text="Invalid numeric inputs! Check your inputs.", text_color="red")
            return

        # Convert inputs to appropriate data types
        age = int(age)
        income = float(income)

        # Map categorical inputs to numeric values
        gender_map = {'Female': 0, 'Male': 1}
        education_map = {'Bachelor': 0, 'Master': 1, 'High School': 2, 'Associate': 3, 'Doctoral': 4}
        marital_map = {'Single': 0, 'Married': 1}

        gender_value = gender_map.get(gender, -1)
        education_value = education_map.get(education_level, -1)
        marital_value = marital_map.get(marital_status, -1)

        # Validate mapped values
        if gender_value == -1 or education_value == -1 or marital_value == -1:
            result_label.configure(text="Invalid categorical inputs! Check your inputs.", text_color="red")
            return

        # One-hot encoding for occupation
        occupation_encoded = [1 if occ == occupation else 0 for occ in all_occupations]

        # Prepare feature array
        input_data = pd.DataFrame([[age, gender_value] + occupation_encoded + [education_value, marital_value, income]],
                                  columns=feature_columns)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict using the loaded model
        prediction = model.predict(input_data_scaled)

        # Update result in the UI
        if prediction[0] == 1:
            result_label.configure(text="Loan Approved", text_color="green")
        else:
            result_label.configure(text="Loan Denied", text_color="red")

    except Exception as e:
        result_label.configure(text=f"Error: {e}", text_color="red")


# Create GUI using customtkinter
root = ctk.CTk()
root.title("Loan Approval Prediction")
root.geometry("1000x900")

# Add header label at the top
header_label = ctk.CTkLabel(root, text="Loan Approval Prediction", font=("Verdana", 20, "bold"))
header_label.pack(pady=20)

# Input Frame
# Input Frame
input_frame = ctk.CTkFrame(root, corner_radius=10)
input_frame.pack(pady=20, padx=20, fill="both", expand=True)

# Center alignment for the form within the input_frame
input_frame.grid_columnconfigure(0, weight=1)  # Center-align column 0
input_frame.grid_columnconfigure(1, weight=1)  # Center-align column 1

# Input Fields
ctk.CTkLabel(input_frame, text="Age:", font=("Verdana", 16)).grid(row=0, column=0, padx=10, pady=10, sticky="e")
age_entry = ctk.CTkEntry(input_frame, font=("Verdana", 16))
age_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")

ctk.CTkLabel(input_frame, text="Gender:", font=("Verdana", 16)).grid(row=1, column=0, padx=10, pady=10, sticky="e")
gender_combobox = ctk.CTkComboBox(input_frame, values=["Female", "Male"], font=("Verdana", 16))
gender_combobox.grid(row=1, column=1, padx=10, pady=10, sticky="w")

ctk.CTkLabel(input_frame, text="Income:", font=("Verdana", 16)).grid(row=2, column=0, padx=10, pady=10, sticky="e")
income_entry = ctk.CTkEntry(input_frame, font=("Verdana", 16))
income_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

ctk.CTkLabel(input_frame, text="Education Level:", font=("Verdana", 16)).grid(row=3, column=0, padx=10, pady=10, sticky="e")
education_combobox = ctk.CTkComboBox(input_frame, values=["Bachelor", "Master", "High School", "Associate", "Doctoral"], font=("Verdana", 16))
education_combobox.grid(row=3, column=1, padx=10, pady=10, sticky="w")

ctk.CTkLabel(input_frame, text="Marital Status:", font=("Verdana", 16)).grid(row=4, column=0, padx=10, pady=10, sticky="e")
marital_combobox = ctk.CTkComboBox(input_frame, values=["Single", "Married"], font=("Verdana", 16))
marital_combobox.grid(row=4, column=1, padx=10, pady=10, sticky="w")

ctk.CTkLabel(input_frame, text="Occupation:", font=("Verdana", 16)).grid(row=5, column=0, padx=10, pady=10, sticky="e")
occupation_combobox = ctk.CTkComboBox(input_frame, values=all_occupations, font=("Verdana", 16))
occupation_combobox.grid(row=5, column=1, padx=10, pady=10, sticky="w")

# Predict Button
predict_button = ctk.CTkButton(
    root,
    text="Predict Loan Status",
    corner_radius=20,
    font=("Verdana", 16),
    command=predict_loan_status,
    fg_color="#000000",  # พื้นหลังปุ่มสีดำ
    text_color="#FFFFFF",  # ข้อความสีขาว
    hover_color="#555555"  # สีเทาเมื่อเมาส์เลื่อนผ่าน
)
predict_button.pack(pady=20)

# Result Label
result_label = ctk.CTkLabel(
    root,
    text="",  # เริ่มต้นด้วยข้อความว่าง
    font=("Verdana", 16),
    text_color="#FF0000"  # กำหนดสีเริ่มต้นเป็นสีแดง
)
result_label.pack(pady=10)

# Data Table Frame
table_frame = ctk.CTkFrame(root, corner_radius=10)
table_frame.pack(pady=20, padx=20, fill="both", expand=True)

# Treeview for Data Table
tree = ttk.Treeview(table_frame, columns=list(data.columns), show="headings", height=15)
tree.pack(fill="both", expand=True)

# Configure Treeview Style
style = ttk.Style(root)
style.theme_use("clam")

# กำหนดสไตล์หัวตาราง
style.configure(
    "Treeview.Heading",
    background="black",
    foreground="white",
    font=("Verdana", 12, "bold")
)

# กำหนดสีแถวและข้อความ
style.configure(
    "Treeview",
    background="white",
    foreground="black",
    rowheight=25,
    fieldbackground="white",
    font=("Verdana", 10)
)

# แสดงสีแถวสลับกัน (Striped Rows)
tree.tag_configure("oddrow", background="#E8E8E8")  # สีเทาอ่อน
tree.tag_configure("evenrow", background="#FFFFFF")  # สีขาว

# Add Table Headers
for col in data.columns:
    tree.heading(col, text=col)
    tree.column(col, anchor="center", width=150)  # กำหนดความกว้างของคอลัมน์

# Add Data to Table with striped rows
for index, row in data.iterrows():
    tag = "evenrow" if index % 2 == 0 else "oddrow"
    tree.insert("", "end", values=list(row), tags=(tag,))

root.mainloop()
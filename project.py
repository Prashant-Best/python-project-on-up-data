import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
raw = pd.read_csv('UP DATA.csv', low_memory=False)


# Data overview
print("=== HEAD ===")
print(raw.head(), "\n")

print("=== TAIL ===")
print(raw.tail(), "\n")

print("=== INFO ===")
print(raw.info(), "\n")

print("=== DESCRIPTION ===")
print(raw.describe(include='all'), "\n")

print("=== NULL VALUES PER COLUMN BEFORE ANY CLEANING ===")
print(raw.isnull().sum(), "\n")


# Changing null values to its mean



raw["AuthorizedCapital"] = raw["AuthorizedCapital"].fillna(raw["AuthorizedCapital"].mean())
raw["PaidupCapital"] = raw["PaidupCapital"].fillna(raw["PaidupCapital"].mean())

print("=== NULL VALUES PER COLUMN  AFTER CLEAING 2 COLUMS CLEANING ===")
print(raw.isnull().sum(), "\n")





# Drop null values and work on a clean copy
df = raw.dropna().copy()

print("=== NULL VALUES PER COLUMN  AFTER CLEAING  CLEANING ===")
print(df.isnull().sum(), "\n")



# Convert registration date to datetime format (dd/mm/yy)
df["CompanyRegistrationdate_date"] = pd.to_datetime(
    df["CompanyRegistrationdate_date"],
    format="%d/%m/%y",
    errors="coerce"
)



# Drop rows with invalid dates
df = df[df["CompanyRegistrationdate_date"].notnull()].copy()



# Extract registration year
df["RegistrationYear"] = df["CompanyRegistrationdate_date"].dt.year


# Plot 1: Company Class Distribution
sns.countplot(
    data=df,
    x="CompanyClass",
    order=df["CompanyClass"].value_counts().index,
    hue="CompanyClass",
    palette="pastel",
    legend=False
)
plt.title("Company Class Distribution")
plt.xticks(rotation=45)
plt.show()



# Plot 2: Company Status Distribution
sns.countplot(
    data=df,
    y="CompanyStatus",
    order=df["CompanyStatus"].value_counts().index,
    hue="CompanyStatus",
    palette="muted",
    legend=False
)
plt.title("Company Status Distribution")
plt.show()




# Plot 3: Capital Distribution (log scale)
sns.boxplot(data=df[["AuthorizedCapital", "PaidupCapital"]], palette="Set2")
plt.yscale("log")
plt.title("Capital Distribution (Log Scale)")
plt.show()



# Plot 3.1: Capital Distribution (After outliner removal)
Q1_AC = df["AuthorizedCapital"].quantile(0.25)
Q3_AC = df["AuthorizedCapital"].quantile(0.75)
IQR_AC = Q3_AC - Q1_AC

df = df[
    (df["AuthorizedCapital"] >= (Q1_AC - 1.5 * IQR_AC)) & 
    (df["AuthorizedCapital"] <= (Q3_AC + 1.5 * IQR_AC))
]



# Remove outliers for PaidupCapital
Q1_PC = df["PaidupCapital"].quantile(0.25)
Q3_PC = df["PaidupCapital"].quantile(0.75)
IQR_PC = Q3_PC - Q1_PC

df = df[
    (df["PaidupCapital"] >= (Q1_PC - 1.5 * IQR_PC)) & 
    (df["PaidupCapital"] <= (Q3_PC + 1.5 * IQR_PC))
]


sns.boxplot(data=df[["AuthorizedCapital", "PaidupCapital"]], palette="Set2")
plt.yscale("log")
plt.title("Capital Distribution (Log Scale) After Removing Outliers")
plt.show()



# Plot 4: Top 10 Industrial Classifications
top_industries = df["CompanyIndustrialClassification"].value_counts().nlargest(10)
sns.barplot(
    y=top_industries.index,
    x=top_industries.values,
    hue=top_industries.index,
    palette="viridis",
    legend=False
)
plt.title("Top 10 Industrial Classifications")
plt.xlabel("Number of Companies")
plt.show()



# Plot 5: Company Registrations Over the Years
reg_years = df["RegistrationYear"].value_counts().sort_index()
sns.lineplot(x=reg_years.index, y=reg_years.values, marker="o", color="coral")
plt.title("Company Registrations Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Companies")
plt.show()

# Plot 6: Distribution of Authorized Capital (Histogram)
sns.histplot(df["AuthorizedCapital"], bins=50, color='skyblue', log_scale=True)
plt.title("Authorized Capital Distribution (Histogram)")
plt.xlabel("Authorized Capital")
plt.show()

# Plot 7: Full Correlation Heatmap of Numerical Features
plt.figure(figsize=(10, 6))
corr = df.select_dtypes(include=["number"]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

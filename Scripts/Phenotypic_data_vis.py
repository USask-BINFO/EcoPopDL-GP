# STEP I

## Phenotypic data visualization

# i. Yield D1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("WCC_Yield.csv", delimiter=',')  # Adjust the delimiter if needed

# Convert yield columns to numeric, handling missing values
df['Yield_SD1'] = pd.to_numeric(df['Yield_SD1'], errors='coerce')
df['Yield_SD2'] = pd.to_numeric(df['Yield_SD2'], errors='coerce')

# Melt the DataFrame to long format for plotting
df_long = pd.melt(df, id_vars=["Name", "Year/Location"], value_vars=['Yield_SD1', 'Yield_SD2'],
                  var_name="Yield_Type", value_name="Yield_Value")

# Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=df_long, x="Year/Location", y="Yield_Value", hue="Yield_Type", split=True)

# Calculate mean values to display on the plot
mean_values = df_long.groupby(["Year/Location", "Yield_Type"])['Yield_Value'].mean().reset_index()

# Add mean values as text slightly above each violin plot
for index, row in mean_values.iterrows():
    x_pos = row["Year/Location"]  # Category for Year/Location on x-axis
    y_pos = row["Yield_Value"]
    
    # Determine horizontal offset based on Yield_Type
    x_offset = -0.25 if row["Yield_Type"] == "Yield_SD1" else 0.25
    y_offset = 110  # Adjust this value to move text higher or lower

    # Place the text at the specified position
    plt.text(
        x=plt.xticks()[0][list(df_long['Year/Location'].unique()).index(x_pos)] + x_offset, 
        y=y_pos + y_offset,  # Move text slightly up by adding y_offset
        s=f"{row['Yield_Value']:.1f}", 
        color="red", 
        ha='center', 
        fontsize=20, 
        weight="bold"
    )

plt.xlabel("Year/Location", fontsize=14)
plt.ylabel("Yield Value", fontsize=14)
plt.title("Distribution of Yield Across Different Locations and Yield Types", fontsize=14)
plt.legend(title="Yield Type", fontsize=14)
# Increase font size of x-axis tick labels
plt.xticks(fontsize=12)  # Adjust fontsize here as desired
plt.yticks(fontsize=12)  # Adjust fontsize here as desired
plt.tight_layout()
plt.show()



# ii. DTF D1


# Load the dataset
df = pd.read_csv("WCC_Flowering.csv", delimiter=',')  # Adjust the delimiter if needed

# Melt the DataFrame to long format for easier plotting
df_long = pd.melt(df, id_vars=["Name"], value_vars=["MJ2019", "LL2019", "LL2020"],
                  var_name="Location", value_name="DTF_Value")

# Violin Plot with Custom Colors
plt.figure(figsize=(10, 6))
colors = {"MJ2019": "green", "LL2019": "orange", "LL2020": "gray"}
sns.violinplot(data=df_long, x="Location", y="DTF_Value", palette=colors)

# Adding labels and title
plt.xlabel("Location", fontsize=14)
plt.ylabel("DTF Value", fontsize=14)
plt.title("Distribution of Days to Flowering (DTF) Values Across Different Locations (SD1)", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# ii. SW D1



# Load the dataset
df = pd.read_csv("WCC_SW.csv", delimiter=',')  # Adjust the delimiter if needed

# Convert yield columns to numeric, handling missing values
df['1000 SW_SD1'] = pd.to_numeric(df['1000 SW_SD1'], errors='coerce')
df['1000 SW_SD2'] = pd.to_numeric(df['1001 SW_SD2'], errors='coerce')  # Corrected column name

# Melt the DataFrame to long format for plotting
df_long = pd.melt(df, id_vars=["Name", "Year/Location"], value_vars=['1000 SW_SD1', '1000 SW_SD2'],
                  var_name="Yield_Type", value_name="Yield_Value")

# Violin Plot with Custom Colors
plt.figure(figsize=(12, 6))
sns.violinplot(
    data=df_long, 
    x="Year/Location", 
    y="Yield_Value", 
    hue="Yield_Type", 
    split=True,
    palette={"1000 SW_SD1": "green", "1000 SW_SD2": "purple"}  # New colors for each type
)

# Calculate mean values to display on the plot
mean_values = df_long.groupby(["Year/Location", "Yield_Type"])['Yield_Value'].mean().reset_index()

# Add mean values as text slightly above each violin plot
for index, row in mean_values.iterrows():
    x_pos = row["Year/Location"]  # Category for Year/Location on x-axis
    y_pos = row["Yield_Value"]
    
    # Determine horizontal offset based on Yield_Type
    x_offset = -0.25 if row["Yield_Type"] == "1000 SW_SD1" else 0.25
    y_offset = 110  # Adjust this value to move text higher or lower

    # Place the text at the specified position
    plt.text(
        x=plt.xticks()[0][list(df_long['Year/Location'].unique()).index(x_pos)] + x_offset, 
        y=y_pos + y_offset,  # Move text slightly up by adding y_offset
        s=f"{row['Yield_Value']:.1f}", 
        color="red", 
        ha='center', 
        fontsize=20, 
        weight="bold"
    )

# Adjust labels and title font sizes
plt.xlabel("Year/Location", fontsize=14)
plt.ylabel("Seed Size", fontsize=14)
plt.title("Distribution of Seed Size Across Different Locations and Seed Weight Types", fontsize=14)
plt.legend(title="SW Type", fontsize=14)
plt.xticks(fontsize=12)  # Adjust x-axis tick label font size
plt.yticks(fontsize=12)  # Adjust y-axis tick label font size
plt.tight_layout()
plt.show()

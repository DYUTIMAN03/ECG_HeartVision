import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
csv_path = 'data/raw_dataset/MIT-BIH.csv'  # Update this path if needed
output_dir = 'data/images'
normal_dir = os.path.join(output_dir, 'normal')
abnormal_dir = os.path.join(output_dir, 'abnormal')

# Create output dirs
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(abnormal_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(csv_path)
print("Total samples in original dataset:", len(df))
print(df['target'].value_counts())  # Show label distribution

# Set label column name
label_col = 'target'

# Separate normal (0) and abnormal (1–4)
normal_df_all = df[df[label_col] == 0]
abnormal_df_all = df[df[label_col] != 0]

# Match both classes to the smallest available count
normal_count = len(normal_df_all)  # Should be 779
abnormal_count = normal_count      # Balance both classes

print(f"\nUsing {normal_count} samples from each class (normal & abnormal)")

# Sample the data
normal_df = normal_df_all
abnormal_df = abnormal_df_all.sample(n=abnormal_count, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([normal_df, abnormal_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save each sample as an image
for i, row in balanced_df.iterrows():
    signal = row.drop(['Unnamed: 0', label_col]).values.astype(float)
    label = 'normal' if row[label_col] == 0 else 'abnormal'
    save_dir = normal_dir if label == 'normal' else abnormal_dir
    save_path = os.path.join(save_dir, f"{label}_{i}.png")

    # Plot ECG signal and save as image
    plt.figure(figsize=(2, 2))
    plt.plot(signal, color='black')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"\n✅ Saved {normal_count} normal and {abnormal_count} abnormal images to '{output_dir}'")

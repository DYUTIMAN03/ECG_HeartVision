import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Example: Dummy prediction probabilities for 10 ECG images (2 classes: abnormal, normal)
# Replace this with real prediction probabilities from your model
pred_probs = np.array([
    [0.8, 0.2],
    [0.1, 0.9],
    [0.7, 0.3],
    [0.6, 0.4],
    [0.2, 0.8],
    [0.9, 0.1],
    [0.3, 0.7],
    [0.4, 0.6],
    [0.5, 0.5],
    [0.85, 0.15]
])

# Convert to DataFrame for correlation analysis
df_probs = pd.DataFrame(pred_probs, columns=['abnormal', 'normal'])

# Calculate correlation matrix
correlation_matrix = df_probs.corr()

# Plot heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Prediction Probabilities')
plt.show()

import pandas as pd
import numpy as np

# A list of common crops and their ideal conditions.
# Note: These are simplified ideal ranges for generating plausible synthetic data.
crop_conditions = {
    'Rice': {'N': (70, 90), 'P': (40, 60), 'K': (40, 60), 'temperature': (25, 35), 'humidity': (70, 90), 'ph': (5.5, 7.0), 'rainfall': (200, 300)},
    'Maize': {'N': (75, 95), 'P': (40, 60), 'K': (20, 40), 'temperature': (20, 30), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (60, 100)},
    'Chickpea': {'N': (15, 30), 'P': (30, 50), 'K': (70, 90), 'temperature': (15, 20), 'humidity': (40, 60), 'ph': (7.0, 8.0), 'rainfall': (60, 80)},
    'KidneyBeans': {'N': (20, 40), 'P': (60, 80), 'K': (15, 25), 'temperature': (20, 25), 'humidity': (50, 70), 'ph': (5.5, 6.5), 'rainfall': (90, 120)},
    'PigeonPeas': {'N': (20, 40), 'P': (60, 80), 'K': (15, 25), 'temperature': (20, 25), 'humidity': (50, 70), 'ph': (5.5, 6.5), 'rainfall': (90, 120)},
    'MothBeans': {'N': (20, 40), 'P': (60, 80), 'K': (15, 25), 'temperature': (25, 35), 'humidity': (40, 60), 'ph': (6.0, 7.0), 'rainfall': (30, 50)},
    'MungBean': {'N': (20, 40), 'P': (60, 80), 'K': (15, 25), 'temperature': (20, 25), 'humidity': (50, 70), 'ph': (5.5, 6.5), 'rainfall': (90, 120)},
    'Blackgram': {'N': (20, 40), 'P': (60, 80), 'K': (15, 25), 'temperature': (25, 35), 'humidity': (40, 60), 'ph': (6.0, 7.0), 'rainfall': (30, 50)},
    'Lentil': {'N': (20, 40), 'P': (60, 80), 'K': (15, 25), 'temperature': (15, 20), 'humidity': (40, 60), 'ph': (6.0, 7.0), 'rainfall': (30, 50)},
    'Pomegranate': {'N': (30, 50), 'P': (60, 80), 'K': (30, 50), 'temperature': (20, 25), 'humidity': (50, 60), 'ph': (5.5, 7.0), 'rainfall': (40, 60)},
    'Banana': {'N': (80, 100), 'P': (70, 90), 'K': (50, 70), 'temperature': (25, 35), 'humidity': (80, 90), 'ph': (5.5, 6.5), 'rainfall': (100, 150)},
    'Mango': {'N': (30, 50), 'P': (40, 60), 'K': (20, 40), 'temperature': (25, 35), 'humidity': (60, 80), 'ph': (5.0, 6.5), 'rainfall': (100, 150)},
    'Grapes': {'N': (10, 20), 'P': (20, 30), 'K': (10, 20), 'temperature': (20, 25), 'humidity': (60, 70), 'ph': (6.0, 7.0), 'rainfall': (50, 70)},
    'Watermelon': {'N': (15, 25), 'P': (25, 35), 'K': (15, 25), 'temperature': (25, 35), 'humidity': (70, 80), 'ph': (6.0, 7.0), 'rainfall': (80, 120)},
    'Muskmelon': {'N': (15, 25), 'P': (25, 35), 'K': (15, 25), 'temperature': (25, 35), 'humidity': (70, 80), 'ph': (6.0, 7.0), 'rainfall': (80, 120)},
    'Apple': {'N': (15, 25), 'P': (25, 35), 'K': (15, 25), 'temperature': (10, 15), 'humidity': (70, 80), 'ph': (6.0, 7.0), 'rainfall': (80, 120)},
    'Orange': {'N': (20, 30), 'P': (10, 20), 'K': (10, 20), 'temperature': (25, 30), 'humidity': (60, 70), 'ph': (6.0, 7.0), 'rainfall': (50, 70)},
    'Papaya': {'N': (20, 30), 'P': (10, 20), 'K': (10, 20), 'temperature': (20, 25), 'humidity': (70, 80), 'ph': (5.5, 6.5), 'rainfall': (80, 120)},
    'Coconut': {'N': (20, 30), 'P': (10, 20), 'K': (10, 20), 'temperature': (25, 30), 'humidity': (80, 90), 'ph': (6.0, 7.0), 'rainfall': (150, 200)},
    'Cotton': {'N': (20, 40), 'P': (20, 40), 'K': (20, 40), 'temperature': (25, 35), 'humidity': (50, 70), 'ph': (6.0, 7.0), 'rainfall': (50, 70)},
    'Jute': {'N': (30, 50), 'P': (10, 20), 'K': (10, 20), 'temperature': (25, 35), 'humidity': (80, 90), 'ph': (6.0, 7.0), 'rainfall': (200, 300)},
    'Coffee': {'N': (20, 40), 'P': (20, 40), 'K': (20, 40), 'temperature': (20, 30), 'humidity': (70, 80), 'ph': (5.5, 6.5), 'rainfall': (100, 150)},
    'Tea': {'N': (20, 40), 'P': (20, 40), 'K': (20, 40), 'temperature': (20, 30), 'humidity': (70, 80), 'ph': (4.5, 5.5), 'rainfall': (150, 200)}
}

# Number of synthetic data points
num_samples = 10000

# Create synthetic dataset
data = []
for _ in range(num_samples):
    crop = np.random.choice(list(crop_conditions.keys()))
    conditions = crop_conditions[crop]

    N = np.random.uniform(*conditions['N'])
    P = np.random.uniform(*conditions['P'])
    K = np.random.uniform(*conditions['K'])
    temperature = np.random.uniform(*conditions['temperature'])
    humidity = np.random.uniform(*conditions['humidity'])
    ph = np.random.uniform(*conditions['ph'])
    rainfall = np.random.uniform(*conditions['rainfall'])

    data.append([N, P, K, temperature, humidity, ph, rainfall, crop])

# Create DataFrame
columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('crop_data.csv', index=False)

print(f"✅ Successfully generated a dataset of {num_samples} rows and saved it to 'crop_data.csv'")
import pandas as pd
import numpy as np
import random

def generate_contextual_crop_data(num_samples: int = 10000) -> pd.DataFrame:
    crop_data = {
        'Rice': {'season': 'Kharif', 'N': (70, 90), 'P': (40, 60), 'K': (40, 60), 'ph': (5.5, 7.0), 'humidity': (70, 90), 'temp_range': (25, 35), 'rain_range': (200, 300)},
        'Maize': {'season': 'Kharif', 'N': (75, 95), 'P': (40, 60), 'K': (20, 40), 'ph': (6.0, 7.5), 'humidity': (60, 80), 'temp_range': (20, 30), 'rain_range': (60, 100)},
        'MungBean': {'season': 'Zaid', 'N': (20, 40), 'P': (60, 80), 'K': (15, 25), 'ph': (5.5, 6.5), 'humidity': (50, 70), 'temp_range': (20, 25), 'rain_range': (90, 120)},
        'PigeonPeas': {'season': 'Zaid', 'N': (20, 40), 'P': (60, 80), 'K': (15, 25), 'ph': (5.5, 6.5), 'humidity': (50, 70), 'temp_range': (20, 25), 'rain_range': (90, 120)},
        'Wheat': {'season': 'Rabi', 'N': (75, 95), 'P': (40, 60), 'K': (20, 40), 'ph': (6.0, 7.5), 'humidity': (60, 80), 'temp_range': (10, 20), 'rain_range': (10, 50)},
        'Chickpea': {'season': 'Rabi', 'N': (15, 30), 'P': (30, 50), 'K': (70, 90), 'ph': (7.0, 8.0), 'humidity': (40, 60), 'temp_range': (15, 20), 'rain_range': (60, 80)},
        'Mustard': {'season': 'Rabi', 'N': (20, 40), 'P': (20, 40), 'K': (20, 40), 'ph': (6.0, 7.0), 'humidity': (50, 70), 'temp_range': (15, 25), 'rain_range': (20, 40)},
        'Sugarcane': {'season': 'Kharif', 'N': (70, 90), 'P': (40, 60), 'K': (40, 60), 'ph': (6.0, 8.0), 'humidity': (70, 90), 'temp_range': (25, 35), 'rain_range': (150, 250)},
    }

    rotation_probabilities = {
        'Wheat': {'Rice': 0.70, 'Maize': 0.15, 'MungBean': 0.10, 'PigeonPeas': 0.05},
        'Rice': {'Wheat': 0.40, 'Maize': 0.40, 'Fallow': 0.20},
        'Maize': {'Wheat': 0.60, 'Rice': 0.40},
        'MungBean': {'Wheat': 0.60, 'Rice': 0.40},
        'PigeonPeas': {'Wheat': 0.50, 'Rice': 0.50},
        'Chickpea': {'Maize': 0.60, 'Millet': 0.40},
        'Mustard': {'Maize': 0.50, 'Wheat': 0.50},
        'Sugarcane': {'Wheat': 0.60, 'Rice': 0.40},
    }

    nutrient_contributions = {
        'MungBean': {'N_boost': 15, 'P_boost': 5},
        'PigeonPeas': {'N_boost': 15, 'P_boost': 5},
        'Fallow': {'N_boost': -10, 'P_boost': -5},
    }

    farm_size_categories = {
        'marginal': (0.1, 1.0),
        'small': (1.0, 2.0),
        'medium': (4.0, 10.0),
        'large': (10.0, 20.0),
    }
    farm_size_weights = [0.86, 0.1, 0.03, 0.01]

    crops = list(crop_data.keys())
    data = []

    for _ in range(num_samples):
        current_crop = random.choice(crops)
        season = crop_data[current_crop]['season']

        if season == 'Kharif':
            temp = random.uniform(28, 35)
            rain = random.uniform(200, 300)
        elif season == 'Rabi':
            temp = random.uniform(10, 20)
            rain = random.uniform(10, 50)
        else:
            temp = random.uniform(25, 30)
            rain = random.uniform(50, 100)

        if current_crop in rotation_probabilities:
            prev_options = list(rotation_probabilities[current_crop].keys())
            prev_probs = list(rotation_probabilities[current_crop].values())
            previous_crop = random.choices(prev_options, prev_probs, k=1)[0]
        else:
            previous_crop = 'Fallow'

        n_min, n_max = crop_data[current_crop]['N']
        p_min, p_max = crop_data[current_crop]['P']
        k_min, k_max = crop_data[current_crop]['K']

        if previous_crop in nutrient_contributions:
            n_boost = nutrient_contributions[previous_crop].get('N_boost', 0)
            p_boost = nutrient_contributions[previous_crop].get('P_boost', 0)
            n_min += n_boost
            n_max += n_boost
            p_min += p_boost
            p_max += p_boost

        N = np.random.uniform(n_min, n_max)
        P = np.random.uniform(p_min, p_max)
        K = np.random.uniform(k_min, k_max)
        ph_min, ph_max = crop_data[current_crop]['ph']
        humidity_min, humidity_max = crop_data[current_crop]['humidity']
        ph = np.random.uniform(ph_min, ph_max)
        humidity = np.random.uniform(humidity_min, humidity_max)

        size_category = random.choices(list(farm_size_categories.keys()), weights=farm_size_weights, k=1)[0]
        min_size, max_size = farm_size_categories[size_category]
        farm_size = round(random.uniform(min_size, max_size), 2)

        data.append([
            round(N, 2), round(P, 2), round(K, 2),
            round(temp, 2), round(humidity, 2), round(ph, 2),
            round(rain, 2), previous_crop, farm_size, current_crop
        ])

    columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'previous_crop', 'farm_size_hectares', 'current_crop']
    df = pd.DataFrame(data, columns=columns)
    return df

if __name__ == '__main__':
    num_data_points = 10000
    df = generate_contextual_crop_data(num_data_points)
    file_name = 'crop_data_contextual.csv'
    df.to_csv(file_name, index=False)
    print(f"Successfully generated a dataset of {num_data_points} rows and saved it to '{file_name}'")

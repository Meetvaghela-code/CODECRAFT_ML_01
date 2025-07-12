import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle
# Load and drop unused columns
data = pd.read_csv(r'data/database.csv')
data.drop(columns=['area_type', 'availability', 'society', 'balcony'], inplace=True)

# Fill missing values
data['size'] = data['size'].fillna('2 BHK')
data['bath'] = data['bath'].fillna(data['bath'].median())

# Extract BHK from size
data['BHK'] = data['size'].str.extract('(\d+)').astype(int)

# Clean total_sqft
def clean_total_sqft(x):
    try:
        tokens = str(x).split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(clean_total_sqft)
data = data.dropna(subset=['total_sqft'])

# Remove unrealistic values
data = data[(data['total_sqft'] / data['BHK']) >= 300]
data = data[data['BHK'] <= 20]

# Add price per sqft column
data['price_pr_sqft'] = data['price'] * 100000 / data['total_sqft']

# Clean and group rare locations
data['location'] = data['location'].fillna('unknown').apply(lambda x: str(x).strip())
loc_counts = data['location'].value_counts()
rare_locs = loc_counts[loc_counts <= 10]
data['location'] = data['location'].apply(lambda x: 'other' if x in rare_locs else x)

# Remove outliers based on price per sqft within each location
def remove_outliers(df):
    df_out = pd.DataFrame()
    for loc, subdf in df.groupby('location'):
        mean = subdf['price_pr_sqft'].mean()
        std = subdf['price_pr_sqft'].std()
        filtered = subdf[(subdf['price_pr_sqft'] >= (mean - std)) & (subdf['price_pr_sqft'] <= (mean + std))]
        df_out = pd.concat([df_out, filtered], ignore_index=True)
    return df_out

data = remove_outliers(data)

# Remove outliers where a higher BHK is priced lower than smaller BHK in same location
def remove_bhk_outliers(df):
    exclude_idx = np.array([])
    for loc, loc_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in loc_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': bhk_df['price_pr_sqft'].mean(),
                'std': bhk_df['price_pr_sqft'].std(),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in loc_df.groupby('BHK'):
            prev_stats = bhk_stats.get(bhk - 1)
            if prev_stats and prev_stats['count'] > 5:
                bad_idx = bhk_df[bhk_df['price_pr_sqft'] < prev_stats['mean']].index.values
                exclude_idx = np.append(exclude_idx, bad_idx)
    return df.drop(exclude_idx, axis='index')

data = remove_bhk_outliers(data)

# Feature engineering
data['total_per_bhk'] = data['total_sqft'] / data['BHK']
data['bath_per_bhk'] = data['bath'] / data['BHK']

# One-hot encode location
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# Drop unnecessary columns
data.drop(columns=['size', 'price_pr_sqft'], inplace=True, errors='ignore')

# Save cleaned data (optional)
data.to_csv('clean_data_model.csv', index=False)

# Separate features and target
X = data[['total_sqft', 'bath', 'BHK', 'total_per_bhk', 'bath_per_bhk']]
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create pipeline with scaling and linear regression
pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipe.predict(X_test)
score = r2_score(y_test, y_pred)
print(f'Linear Regression R2 Score: {score:.4f}')

pickle.dump(pipe,open('LinearModel.pkl','wb'))
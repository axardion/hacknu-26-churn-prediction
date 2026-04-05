import pandas as pd

# Load data
train = pd.read_csv('data/preprocessed/train/train_users_properties.csv')
test = pd.read_csv('data/preprocessed/test/test_users_properties.csv')
purchases = pd.read_csv('data/purchases_train.csv')

# Merge train users with their purchases
merged = train[['user_id', 'subscription_plan']].drop_duplicates().merge(
    purchases[['user_id', 'total_spend']], on='user_id', how='inner'
)

# Calculate mean and standard deviation of total spend per subscription plan
plan_stats = merged.groupby('subscription_plan')['total_spend'].agg(
    avg_total_spend='mean',
    std_total_spend='std'
).reset_index()

# Combine train and test users
df = pd.concat([train, test], ignore_index=True)
df = df[['user_id', 'subscription_plan']].drop_duplicates()

# Merge the calculated statistics back into the main dataframe
df = df.merge(plan_stats, on='subscription_plan', how='left')

# Save to CSV
df.to_csv('data/subscriptions.csv', index=False)
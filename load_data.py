import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(filepath):
    """Load the data from a CSV file."""
    data = pd.read_csv(filepath)
    return data

def clean_and_select_data(data):
    """Clean data and select relevant columns."""
    columns_of_interest = ['price', 'room_type', 'accommodates', 'number_of_reviews', 'neighbourhood_cleansed']
    data_selected = data[columns_of_interest]
    data_selected.loc[:, 'price'] = data_selected['price'].replace(r'[\$,]', '', regex=True).astype(float)
    return data_selected

def impute_missing_prices(data):
    """Impute missing prices based on median prices grouped by room type and neighbourhood."""
    price_imputation = data.groupby(['room_type', 'neighbourhood_cleansed'])['price'].median()
    data['price'] = data.apply(
        lambda row: price_imputation[(row['room_type'], row['neighbourhood_cleansed'])] if pd.isnull(row['price']) else row['price'],
        axis=1
    )
    return data

def create_visualizations(data):
    """Generate and save visualizations to analyze pricing trends."""
    sns.set(style="whitegrid")

    # Plot 1: Price Distribution by Room Type
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='room_type', y='price', data=data)
    plt.title('Price Distribution by Room Type')
    plt.xlabel('Room Type')
    plt.ylabel('Price ($)')
    plt.savefig('price_distribution_by_room_type.png')  # Save Plot 1
    plt.close()

    # Plot 2: Price Trends by Number of Accommodates
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='accommodates', y='price', data=data)
    plt.title('Price Trends by Number of Accommodates')
    plt.xlabel('Accommodates')
    plt.ylabel('Price ($)')
    plt.savefig('price_trends_by_accommodates.png')  # Save Plot 2
    plt.close()

    # Plot 3: Price Distribution by Neighbourhood
    plt.figure(figsize=(10, 8))
    top_neighbourhoods = data['neighbourhood_cleansed'].value_counts().nlargest(10).index
    filtered_data = data[data['neighbourhood_cleansed'].isin(top_neighbourhoods)]
    sns.boxplot(x='price', y='neighbourhood_cleansed', data=filtered_data)
    plt.title('Price Distribution by Neighbourhood')
    plt.xlabel('Price ($)')
    plt.ylabel('Neighbourhood')
    plt.savefig('price_distribution_by_neighbourhood.png')  # Save Plot 3
    plt.close()

    # Plot 4: Relationship between Price and Number of Reviews
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='number_of_reviews', y='price', data=data)
    plt.title('Relationship between Price and Number of Reviews')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Price ($)')
    plt.savefig('price_relationship_number_of_reviews.png')  # Save Plot 4
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # Adjust size as needed

    # create merged plots
    sns.boxplot(x='room_type', y='price', data=data, ax=axs[0, 0])
    axs[0, 0].set_title('Price Distribution by Room Type')

    # Plot 2: Price Trends by Number of Accommodates
    sns.scatterplot(x='accommodates', y='price', data=data, ax=axs[0, 1])
    axs[0, 1].set_title('Price Trends by Number of Accommodates')

    # Assuming you might have two more plots to add here
    # For demonstration, reusing plot 1
    sns.boxplot(x='room_type', y='number_of_reviews', data=data, ax=axs[1, 0])
    axs[1, 0].set_title('Reviews Distribution by Room Type')

    # Reusing plot 2
    sns.scatterplot(x='number_of_reviews', y='price', data=data, ax=axs[1, 1])
    axs[1, 1].set_title('Price vs. Reviews')

    plt.tight_layout()
    plt.savefig('combined_plots.png')  # Save the figure to a file
    plt.close()

def prepare_features(data):
    """Prepare features for regression."""
    # Assuming 'room_type' is categorical and needs encoding
    ct = ColumnTransformer([('one_hot', OneHotEncoder(), ['room_type'])], remainder='passthrough')
    features = ct.fit_transform(data[['room_type', 'accommodates', 'number_of_reviews']])
    return features, data['price']

def train_and_evaluate(features, price):
    """Train the linear regression model and evaluate it."""
    X_train, X_test, y_train, y_test = train_test_split(features, price, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2_score(y_test, y_pred))

def main():
    filepath = 'listings.csv'
    data = load_data(filepath)
    data_selected = clean_and_select_data(data)
    data_imputed = impute_missing_prices(data_selected)
    features, price = prepare_features(data_imputed)
    train_and_evaluate(features, price)
    create_visualizations(data_imputed)

if __name__ == "__main__":
    main()

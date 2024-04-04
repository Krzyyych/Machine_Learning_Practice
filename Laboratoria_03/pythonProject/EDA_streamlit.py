import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns

def read_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    return df

def basic_info(df):
    st.subheader('Dane - podstawowe info')
    if st.checkbox('Wyświetl 5 pierwszych rekordów danych'):
        st.write(df.head())
    if st.checkbox("Wyświetl podstawowe statystyki"):
        st.write(df.describe())

def removing_outliers(df):
    st.subheader("Usuwanie wartości odstających")
    outliers_method = st.selectbox("Wybierz metodę usuwania wartości odstających:",
                                   ['Percentyle', 'Z-score', 'Współczynnik IQR'])
    if outliers_method == 'Percentyle':
        lower_percentile = st.slider("Dolny percentyl:", min_value=0, max_value=50, value=1)
        upper_percentile = st.slider("Górny percentyl:", min_value=50, max_value=100, value=99)
        for feature in df.columns:
            if feature != 'PRICE':  # Ignoruj kolumnę 'PRICE'
                df[feature] = remove_outliers_percentile(df[feature], lower_percentile, upper_percentile)
    elif outliers_method == 'Z-score':
        threshold = st.slider("Próg z-score:", min_value=1, max_value=10, value=3)
        for feature in df.columns:
            if feature != 'PRICE':  # Ignoruj kolumnę 'PRICE'
                df[feature] = remove_outliers_zscore(df[feature], threshold)
    else:
        for feature in df.columns:
            if feature != 'PRICE':  # Ignoruj kolumnę 'PRICE'
                df[feature] = remove_outliers_iqr(df[feature])

def display_hist(df):
    st.subheader('Histogram wybranej cechy')
    selected_feature = st.selectbox('Wybierz cechę do wyświetlenia', df.columns[:-1])
    fig = plt.figure(figsize=(8, 6))
    sns.histplot(df[selected_feature], bins=20, alpha=0.7, kde=True)
    plt.xlabel(selected_feature)
    plt.ylabel('Liczebność')
    st.pyplot(fig)

def display_corr_matrix(df):
    st.subheader('Macierz korelacji')
    fig = plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(fig)
    return corr

def display_scatter(df):
    st.subheader('Wykres punktowy')
    x_feature = st.selectbox('Wybierz cechę do osi X:', df.columns[:-1])
    y_feature = st.selectbox('Wybierz cechę do osi Y:', df.columns[:-1])
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(df[x_feature], df[y_feature])
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f'Wykres punktowy ({x_feature} vs {y_feature})')
    st.pyplot(fig)
    return x_feature, y_feature

def display_corr_value(df, corr, x_feature, y_feature):
    correlation_value = corr.loc[x_feature, y_feature]
    st.write(f'<div style="text-align: center; font-size: 18px; color: blue;">'
             f'Wartość korelacji między <b>{x_feature}</b> a <b>{y_feature}</b>: '
             f'<span style="font-weight: bold; font-size: 20px;">{correlation_value:.2f}</span>'
             '</div>', unsafe_allow_html=True)
def remove_outliers_percentile(data, lower_percentile=1, upper_percentile=99):
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]

def remove_outliers_zscore(data, threshold=3):
    z_scores = (data - data.mean()) / data.std()
    return data[abs(z_scores) < threshold]

def remove_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[(data >= (Q1 - 1.5 * IQR)) & (data <= (Q3 + 1.5 * IQR))]
def main():
    data = read_data()
    st.title('Eksploracyjna Analiza Danych (EDA) - Ceny Mieszkań w Kalifornii')
    basic_info(data)
    removing_outliers(data)
    display_hist(data)
    corr = display_corr_matrix(data)
    x, y = display_scatter(data)
    display_corr_value(data, corr, x, y)

if __name__ == "__main__":
    main()
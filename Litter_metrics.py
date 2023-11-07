import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import scipy.stats as stats
import altair as alt
st.set_page_config(layout="wide")

# # # Load the dataset
# df = pd.read_excel(r'c:\Users\crobbins\Desktop\Polished_liter_metrics.xlsx')

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)

df.loc[(df['Father_geno'] == 'het') | (df['Mother_geno'] == 'het'), 'Cross_het'] = 'het'
df.loc[(df['Father_geno'] == 'wt') | (df['Mother_geno'] == 'wt'), 'Cross_wt'] = 'wt'
df.loc[(df['Father_geno'] == 'ko') | (df['Mother_geno'] == 'ko'), 'Cross_ko'] = 'ko'

df['Cross_het'].fillna(0, inplace=True)
df['Cross_wt'].fillna(0, inplace=True)
df['Cross_ko'].fillna(0, inplace=True)

df.loc[(df['Cross_het'] == 'het') & (df['Cross_wt'] == 0) & (df['Cross_ko'] == 0), 'Cross'] = 'hethet'
df.loc[(df['Cross_het'] == 'het') & (df['Cross_wt'] == 'wt') & (df['Cross_ko'] == 0), 'Cross'] = 'hetwt'
df.loc[(df['Cross_het'] == 'het') & (df['Cross_wt'] == 0) & (df['Cross_ko'] == 'ko'), 'Cross'] = 'hetko'
df.loc[(df['Cross_het'] == 0) & (df['Cross_wt'] == 'wt') & (df['Cross_ko'] == 0), 'Cross'] = 'wtwt'
df.loc[(df['Cross_het'] == 0) & (df['Cross_wt'] == 0) & (df['Cross_ko'] == 'ko'), 'Cross'] = 'koko'


# Make sure that the 'Exp_ratio' column exists
if 'Exp_ratio' not in df.columns:
    df['Exp_ratio'] = None  # Or np.nan, or some other default value

df.loc[df['Cross'] == 'hethet', 'Exp_ratio'] = df.loc[df['Cross'] == 'hethet', 'Exp_ratio'].apply(lambda x: np.array([.25, .5, .25]))
df.loc[df['Cross'] == 'hetwt', 'Exp_ratio'] = df.loc[df['Cross'] == 'hetwt', 'Exp_ratio'].apply(lambda x: np.array([.5, .5, 0]))
df.loc[df['Cross'] == 'hetko', 'Exp_ratio'] = df.loc[df['Cross'] == 'hetko', 'Exp_ratio'].apply(lambda x: np.array([0, .5, .5]))
df.loc[df['Cross'] == 'wtwt', 'Exp_ratio'] = df.loc[df['Cross'] == 'wtwt', 'Exp_ratio'].apply(lambda x: np.array([1, 0, 0]))
df.loc[df['Cross'] == 'koko', 'Exp_ratio'] = df.loc[df['Cross'] == 'koko', 'Exp_ratio'].apply(lambda x: np.array([0, 0, 1]))


df['exp_wt'] = df.apply(lambda row: row['Total_pups'] * row['Exp_ratio'][0], axis=1)
df['exp_het'] = df.apply(lambda row: row['Total_pups'] * row['Exp_ratio'][1], axis=1)
df['exp_ko'] = df.apply(lambda row: row['Total_pups'] * row['Exp_ratio'][2], axis=1)

het_het = df[(df['Father_geno'] == 'het') & (df['Mother_geno'] == 'het')]
het_wt = df[(df['Father_geno'] == 'het') & (df['Mother_geno'] == 'wt')]
wt_het = df[(df['Father_geno'] == 'wt') & (df['Mother_geno'] == 'het')]
ko_het = df[(df['Father_geno'] == 'ko') & (df['Mother_geno'] == 'het')]
het_ko = df[(df['Father_geno'] == 'het') & (df['Mother_geno'] == 'ko')]
ko_wt = df[(df['Father_geno'] == 'ko') & (df['Mother_geno'] == 'wt')]
wt_ko = df[(df['Father_geno'] == 'wt') & (df['Mother_geno'] == 'ko')]
ko_ko = df[(df['Father_geno'] == 'ko') & (df['Mother_geno'] == 'ko')]
pairing_dict = {'het_het' : het_het, 'het_wt' : het_wt, 'wt_het' : wt_het, 'ko_het' : ko_het,
                'het_ko' : het_ko, 'ko_wt' : ko_wt, 'wt_ko' : wt_ko, 'ko_ko' : ko_ko}

obs_dict = {}
for key in pairing_dict:
    obs_dict[key] = np.array([np.sum(pairing_dict[key]['wt']), np.sum(pairing_dict[key]['het']), np.sum(pairing_dict[key]['ko'])])

exp_dict = {'het_het' : np.array([25,50,25]), 'het_wt' : np.array([50,50,0]), 'wt_ht' : np.array([50,50,0]), 
                'het_ko' : np.array([0,50,50]), 'ko_wt' : np.array([0,100,0]), 'wt_ko' : np.array([0,100,0]), 'ko_ko' : np.array([0,0,100])}

total_obs_dict = {}
total_obs_dict['het_wt'] = obs_dict['het_wt'] + obs_dict['wt_het']
# total_obs_dict['ko_wt'] = obs_dict['ko_wt'] + obs_dict['wt_ko']
total_obs_dict['het_ko'] = obs_dict['het_ko'] + obs_dict['ko_het']
total_obs_dict['het_het'] = obs_dict['het_het']

chi2_results = pd.DataFrame()
for key in total_obs_dict:
    filtered_obs = total_obs_dict[key][total_obs_dict[key] != 0]
    filtered_exp_ratio = exp_dict[key][exp_dict[key] != 0]
    filtered_exp = filtered_exp_ratio * np.sum(filtered_obs)/100

    chi2_stat, p_val, dof, ex = stats.chi2_contingency(filtered_obs, filtered_exp)

    # chi2_results.at[key, 'Chi2_stat'], chi2_results.at[key, 'p_val'], chi2_results.at[key, 'dof'], chi2_results.at[key, 'ex'] 



# Create the Streamlit web app
st.title('Litter_metrics')


# CSS styles
st.markdown("""
    <style>
    /* Set the entire page background */
    body {
        background-color: #E8E8E8; /* A light grey background */
    }
    
    /* Style buttons */
    .stButton>button {
        color: white;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        background-color: #4CAF50;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: white;
        color: #4CAF50;
    }
    
    /* Customize the sidebar */
    .css-1d391kg {
        background-color: #f0f2f6;
        color: #333;
    }
    
    /* Sidebar header */
    .css-1v3fvcr {
        background-color: #4CAF50;
        color: white;
    }
    
    /* Style the DataFrame */
    .dataframe th {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
    }
    
    .dataframe td {
        padding: 5px;
    }
    
    .dataframe tbody tr:nth-of-type(odd) {
        background-color: #f2f2f2;
    }
    
    .dataframe tbody tr:hover {
        background-color: #e0e0e0;
    }
    
    /* Typography */
    h1 {
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    
    h2 {
        font-size: 2em;
        font-weight: bold;
        color: #4CAF50;
    }
    
    p, li, .stText, .stTextArea {
        font-size: 1.1em;
        line-height: 1.5;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import plotly.express as px

# Sample data for the DataFrame 'df'
# Assuming 'df' is a DataFrame containing columns: 'Cross', 'Strain', 'wt', 'exp_wt', 'het', 'exp_het', 'ko', 'exp_ko'

# Sidebar filters for Chart 1
with st.sidebar:
    st.header("Filters for Chart 1")
    cross_filter_chart1 = st.multiselect(
        'Select cross for Chart 1',
        options=df['Cross'].unique(),
        default=df['Cross'].unique()
    )
    strain_filter_chart1 = st.multiselect(
        'Select strain for Chart 1',
        options=df['Strain'].unique(),
        default=df['Strain'].unique()
    )

# Apply filters for Chart 1
df_chart1 = df.copy()
if cross_filter_chart1:
    df_chart1 = df_chart1[df_chart1['Cross'].isin(cross_filter_chart1)]
if strain_filter_chart1:
    df_chart1 = df_chart1[df_chart1['Strain'].isin(strain_filter_chart1)]

# Sum data for Chart 1
sum_data_chart1 = {
    'Category': ['wt_obs', 'wt_exp', 'het_obs', 'het_exp', 'ko_obs', 'ko_exp'],
    '#Pups': [
        df_chart1['wt'].sum(),
        df_chart1['exp_wt'].sum(),
        df_chart1['het'].sum(),
        df_chart1['exp_het'].sum(),
        df_chart1['ko'].sum(),
        df_chart1['exp_ko'].sum()
    ]
}
sum_df_chart1 = pd.DataFrame(sum_data_chart1)

# Sidebar filters for Chart 2
with st.sidebar:
    st.header("Filters for Chart 2")
    cross_filter_chart2 = st.multiselect(
        'Select cross for Chart 2',
        options=df['Cross'].unique(),
        default=df['Cross'].unique()
    )
    strain_filter_chart2 = st.multiselect(
        'Select strain for Chart 2',
        options=df['Strain'].unique(),
        default=df['Strain'].unique()
    )

# Apply filters for Chart 2
df_chart2 = df.copy()
if cross_filter_chart2:
    df_chart2 = df_chart2[df_chart2['Cross'].isin(cross_filter_chart2)]
if strain_filter_chart2:
    df_chart2 = df_chart2[df_chart2['Strain'].isin(strain_filter_chart2)]

# Sum data for Chart 2
sum_data_chart2 = {
    'Category': ['wt_obs', 'wt_exp', 'het_obs', 'het_exp', 'ko_obs', 'ko_exp'],
    '#Pups': [
        df_chart2['wt'].sum(),
        df_chart2['exp_wt'].sum(),
        df_chart2['het'].sum(),
        df_chart2['exp_het'].sum(),
        df_chart2['ko'].sum(),
        df_chart2['exp_ko'].sum()
    ]
}
sum_df_chart2 = pd.DataFrame(sum_data_chart2)

# Define colors for the bars
colors = ['#FFD700', '#C0C0C0', '#FFD700', '#C0C0C0', '#FFD700', '#C0C0C0']

# Create the first bar chart for Chart 1 data
fig1 = px.bar(sum_df_chart1, x='Category', y='#Pups', title='Sum of litter numbers for Chart 1')
fig1.update_traces(marker_color=colors)

# Create the second bar chart for Chart 2 data
fig2 = px.bar(sum_df_chart2, x='Category', y='#Pups', title='Sum of litter numbers for Chart 2')
fig2.update_traces(marker_color=colors)


col1, col2 = st.columns(2)  # This creates two columns with equal width

with col1:
    st.plotly_chart(fig1, use_container_width=True)  # Chart 1 will adjust to the column width

with col2:
    st.plotly_chart(fig2, use_container_width=True)  # Chart 2 will adjust to the column width

######################################################################################


sum_data_chart3 = df_chart1.copy()
# Given data
data = {
    'total': np.sum(sum_data_chart3['Total_pups']),
    'wt': np.sum(sum_data_chart3['wt']),
    'het': np.sum(sum_data_chart3['het']),
    'ko': np.sum(sum_data_chart3['ko']),
}

# Convert to DataFrame
chart3_df = pd.DataFrame([data])

# Function to perform Hardy-Weinberg analysis
def hardy_weinberg_analysis(row):
    N = row['total']  # Total number of individuals
    wt = row['wt']
    het = row['het']
    ko = row['ko']

    # Calculate allele frequencies
    p = (2 * wt + het) / (2 * N)  # Frequency of the dominant allele
    q = 1 - p  # Frequency of the recessive allele

    # Expected genotype frequencies under Hardy-Weinberg equilibrium
    expected_wt = p ** 2 * N
    expected_het = 2 * p * q * N
    expected_ko = q ** 2 * N

    return pd.Series([p, q, expected_wt, expected_het, expected_ko], index=['p', 'q', 'expected_wt', 'expected_het', 'expected_ko'])

# Apply the analysis to each row
chart3_df[['p', 'q', 'expected_wt', 'expected_het', 'expected_ko']] = chart3_df.apply(hardy_weinberg_analysis, axis=1)


# Optionally, check for equilibrium
def check_equilibrium(row):
    chi_square = ((row['wt'] - row['expected_wt']) ** 2) / row['expected_wt'] \
                + ((row['het'] - row['expected_het']) ** 2) / row['expected_het'] \
                + ((row['ko'] - row['expected_ko']) ** 2) / row['expected_ko']
    return chi_square < 3.841  # For a 0.05 significance level with 1 degree of freedom

chart3_df['HW_equilibrium'] = chart3_df.apply(check_equilibrium, axis=1)
print(chart3_df[['total', 'wt', 'het', 'ko', 'HW_equilibrium']])

##############################

sum_data_chart4 = df_chart2.copy()
# Given data
data = {
    'total': np.sum(sum_data_chart4['Total_pups']),
    'wt': np.sum(sum_data_chart4['wt']),
    'het': np.sum(sum_data_chart4['het']),
    'ko': np.sum(sum_data_chart4['ko']),
}

# Convert to DataFrame
chart4_df = pd.DataFrame([data])

# Function to perform Hardy-Weinberg analysis
def hardy_weinberg_analysis(row):
    N = row['total']  # Total number of individuals
    wt = row['wt']
    het = row['het']
    ko = row['ko']

    # Calculate allele frequencies
    p = (2 * wt + het) / (2 * N)  # Frequency of the dominant allele
    q = 1 - p  # Frequency of the recessive allele

    # Expected genotype frequencies under Hardy-Weinberg equilibrium
    expected_wt = p ** 2 * N
    expected_het = 2 * p * q * N
    expected_ko = q ** 2 * N

    return pd.Series([p, q, expected_wt, expected_het, expected_ko], index=['p', 'q', 'expected_wt', 'expected_het', 'expected_ko'])

# Apply the analysis to each row
chart4_df[['p', 'q', 'expected_wt', 'expected_het', 'expected_ko']] = chart4_df.apply(hardy_weinberg_analysis, axis=1)

# Optionally, check for equilibrium
def check_equilibrium(row):
    chi_square = ((row['wt'] - row['expected_wt']) ** 2) / row['expected_wt'] \
                + ((row['het'] - row['expected_het']) ** 2) / row['expected_het'] \
                + ((row['ko'] - row['expected_ko']) ** 2) / row['expected_ko']
    return chi_square < 3.841  # For a 0.05 significance level with 1 degree of freedom

chart4_df['HW_equilibrium'] = chart4_df.apply(check_equilibrium, axis=1)
print(chart4_df[['total', 'wt', 'het', 'ko', 'HW_equilibrium']])



col3, col4 = st.columns(2)

with col3:
    st.write(chart3_df)  # This will display the filtered DataFrame in column 3

with col4:
    st.write(chart4_df)  # This will display the filtered DataFrame in column 3


# streamlit run "C:\Users\crobbins\OneDrive - Fred Hutchinson Cancer Research Center\Macros_and_scripts\Python\Litter_metrics.py"
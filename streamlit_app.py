import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import time
import re
from scipy.optimize import curve_fit
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# IMPORTANT: Update this URL if your report generation backend is running elsewhere.
FASTAPI_BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Data Analyst",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- App UI ---
st.markdown("<h1 style='text-align: center;'>AI Data Analyst</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ask questions and generate visualizations from your data.</p>", unsafe_allow_html=True)

# --- Session State for Data, Insights, and Chart ---
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""
if 'insights_text' not in st.session_state:
    st.session_state.insights_text = None
if 'chart_bytes' not in st.session_state:
    st.session_state.chart_bytes = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'uploaded_file_type' not in st.session_state:
    st.session_state.uploaded_file_type = None
if 'report_content' not in st.session_state:
    st.session_state.report_content = None
if 'report_filename' not in st.session_state:
    st.session_state.report_filename = None
if 'report_mime' not in st.session_state:
    st.session_state.report_mime = None
if 'api_key' not in st.session_state:
    # Use environment variable as initial value
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY")

# --- Helper Functions ---
def find_matching_column(df, query_string):
    """
    Finds a column in the DataFrame that matches a query string,
    handling common variations and special characters.
    """
    cleaned_query = re.sub(r'[^a-zA-Z0-9\s]', '', query_string).lower().replace(" ", "")
    for col in df.columns:
        cleaned_col = re.sub(r'[^a-zA-Z0-9\s]', '', col).lower().replace(" ", "")
        if cleaned_col == cleaned_query or cleaned_query in cleaned_col:
            return col
    return None

def get_answer_from_gemini(api_key, df_info, question):
    """
    Sends a prompt to the Gemini API with a comprehensive data summary
    and returns the AI's response.
    """
    if not api_key:
        return "Error: Gemini API key is not set. Please add it to your environment variables or paste it in the sidebar."

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    # Create a more comprehensive data summary
    data_summary = f"""
- Number of rows: {df_info['rows']}
- Columns and their data types:
{df_info['dtypes']}
- First 5 rows of the data:
{df_info['head']}
"""

    full_prompt = f"""You are a data analysis agent. Your task is to analyze the provided data summary and answer the user's question.
**Data Summary:**
{data_summary}

**User's Question:**
{question}

Provide a concise, direct, and well-reasoned answer based *only* on the provided data. Do not make up information. Do not generate any code or code snippets. If the data is insufficient to answer the question, state that.
"""

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": full_prompt}
                ]
            }
        ]
    }

    retries = 0
    max_retries = 3
    delay = 1

    while retries < max_retries:
        try:
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0 and 'content' in result['candidates'][0]:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "The API returned an empty or unexpected response."
        except requests.exceptions.HTTPError as err:
            if err.response.status_code == 429:
                time.sleep(delay)
                delay *= 2
                retries += 1
            elif err.response.status_code == 403:
                return "Error: API request failed with status 403. Check your API key and permissions."
            else:
                return f"HTTP error occurred: {err}"
        except requests.exceptions.RequestException as err:
            return f"An error occurred during the API request: {err}"

    return "Failed to get a response after multiple retries due to rate limiting."

def generate_chart_bytes(df, chart_type, config):
    """Generates a chart image as a bytes object."""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_label = config.get('x_label', '')
    y_label = config.get('y_label', '')
    title = config.get('title', '')
    color = config.get('color', 'skyblue')
    show_grid = config.get('show_grid', False)
    
    try:
        if chart_type == 'bar':
            counts = df[config['col']].value_counts()
            sns.barplot(x=counts.index, y=counts.values, ax=ax, color=color)
            ax.set_title(title or f'Counts of {config["col"]}')
            ax.set_xlabel(x_label or config["col"])
            ax.set_ylabel(y_label or 'Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

        elif chart_type == 'pie':
            counts = df[config['col']].value_counts()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel") if color == 'default' else [color])
            ax.set_title(title or f'Distribution of {config["col"]}')
            ax.axis('equal')

        elif chart_type == 'scatter':
            x_col = config['cols'][0]
            y_col = config['cols'][1]
            sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax, color=color)
            ax.set_title(title or f'Scatter Plot of {x_col} vs {y_col}')
            ax.set_xlabel(x_label or x_col)
            ax.set_ylabel(y_label or y_col)
            plt.tight_layout()
            
        elif chart_type == 'violin':
            # Identify all numerical columns (your gene expression samples)
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found for violin plot.")

            # Melt the DataFrame from wide to long format for plotting
            melted_df = df.melt(value_vars=numeric_cols, var_name='Sample', value_name='Expression')

            sns.violinplot(x='Sample', y='Expression', data=melted_df, ax=ax, palette='viridis')
            ax.set_title(title or 'Violin Plot of Gene Expression by Sample')
            ax.set_xlabel(x_label or 'Sample')
            ax.set_ylabel(y_label or 'Expression Value')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

        elif chart_type == 'box':
            melted_df = df.melt(var_name='Variable', value_name='Value')
            sns.boxplot(x='Variable', y='Value', data=melted_df, ax=ax, color=color)
            ax.set_title(title or 'Box Plot of All Numerical Columns')
            ax.set_xlabel(x_label or 'Variable')
            ax.set_ylabel(y_label or 'Value')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

        elif chart_type == 'heatmap':
            df_for_heatmap = df.copy()
            if 'Unnamed: 0' in df_for_heatmap.columns:
                df_for_heatmap = df_for_heatmap.set_index('Unnamed: 0')
                df_for_heatmap.index.name = None
            
            if all(pd.api.types.is_numeric_dtype(df_for_heatmap[col]) for col in df_for_heatmap.columns):
                sns.heatmap(df_for_heatmap, annot=False, fmt=".2f", cmap="YlGnBu", ax=ax)
            else:
                x_col, y_col = config['cols'][0], config['cols'][1]
                contingency_table = pd.crosstab(df[y_col], df[x_col])
                sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
            
            ax.set_title(title or 'Heatmap')
            ax.set_xlabel(x_label or 'Column')
            ax.set_ylabel(y_label or 'Row')
            plt.tight_layout()
            
        elif chart_type == 'line':
            # --- START of NEW LOGIC for line chart ---
            if 'cols' in config and len(config['cols']) == 2:
                # Two columns specified: use one for x and one for y
                x_col, y_col = config['cols'][0], config['cols'][1]
                # Ensure x-axis is numeric or can be plotted directly, e.g., a time series index
                if pd.api.types.is_numeric_dtype(df[x_col]) or pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    ax.plot(df[x_col], df[y_col], color=color)
                    ax.set_title(title or f'Line Chart of {y_col} vs {x_col}')
                    ax.set_xlabel(x_label or x_col)
                    ax.set_ylabel(y_label or y_col)
                else:
                    # If x_col is not numeric, plot against index
                    ax.plot(df.index, df[y_col], color=color)
                    ax.set_title(title or f'Line Chart of {y_col} over Index')
                    ax.set_xlabel(x_label or 'Index')
                    ax.set_ylabel(y_label or y_col)
            elif 'col' in config:
                # Single column specified: plot it against the DataFrame's index
                y_col = config['col']
                ax.plot(df.index, df[y_col], color=color)
                ax.set_title(title or f'Line Chart of {y_col} over Index')
                ax.set_xlabel(x_label or 'Index')
                ax.set_ylabel(y_label or y_col)
            else:
                # No column specified, raise an error
                raise ValueError("Line chart requires at least one column to be specified.")
            plt.tight_layout()
            # --- END of NEW LOGIC for line chart ---
            
        elif chart_type == 'exponential':
            if 'cols' in config and len(config['cols']) >= 2:
                x_col, y_col = config['cols'][0], config['cols'][1]
            else:
                st.error("Exponential curve requires two numeric columns. Please specify them, e.g., 'show exponential curve of time vs population'")
                plt.close(fig)
                return None
            
            x_data = df[x_col]
            y_data = df[y_col]
            
            if pd.api.types.is_numeric_dtype(x_data) and pd.api.types.is_numeric_dtype(y_data):
                def exponential_func(x, a, b):
                    return a * np.exp(b * x)
                
                ax.scatter(x_data, y_data, color='blue', label='Data Points')
                
                try:
                    params, covariance = curve_fit(exponential_func, x_data, y_data)
                    a, b = params
                    
                    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
                    y_fit = exponential_func(x_fit, a, b)
                    
                    ax.plot(x_fit, y_fit, color='red', label=f'Fitted Curve: $y = {a:.2f} * e^{{{b:.2f}x}}$')
                    
                    ax.set_title(title or f'Exponential Curve of {y_col} vs {x_col}')
                    ax.set_xlabel(x_label or x_col)
                    ax.set_ylabel(y_label or y_col)
                    ax.legend()
                    
                except RuntimeError:
                    st.error("Could not fit an exponential curve to the data. Please check if the data exhibits exponential growth.")
                
            else:
                st.error("Exponential curve requires both columns to be numeric.")
            plt.tight_layout()
        elif chart_type == 'logarithmic':
            if 'cols' in config and len(config['cols']) >= 2:
                x_col, y_col = config['cols'][0], config['cols'][1]
            else:
                st.error("Logarithmic curve requires two numeric columns. Please specify them, e.g., 'show logarithmic curve of time vs population'")
                plt.close(fig)
                return None
            
            x_data = df[x_col]
            y_data = df[y_col]
            
            if (x_data > 0).all() and pd.api.types.is_numeric_dtype(x_data) and pd.api.types.is_numeric_dtype(y_data):
                def log_func(x, a, b):
                    return a * np.log(x) + b
                
                ax.scatter(x_data, y_data, color='blue', label='Data Points')
                
                try:
                    params, covariance = curve_fit(log_func, x_data, y_data)
                    a, b = params
                    
                    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
                    y_fit = log_func(x_fit, a, b)
                    
                    ax.plot(x_fit, y_fit, color='red', label=f'Fitted Curve: $y = {a:.2f} \cdot \ln(x) + {b:.2f}$')
                    
                    ax.set_title(title or f'Logarithmic Curve of {y_col} vs {x_col}')
                    ax.set_xlabel(x_label or x_col)
                    ax.set_ylabel(y_label or y_col)
                    ax.legend()
                    
                except RuntimeError:
                    st.error("Could not fit a logarithmic curve to the data. Please check if the data exhibits a logarithmic trend and that all x-values are positive.")
                
            else:
                st.error("Logarithmic curve requires both columns to be numeric, and all x-values must be positive.")
            plt.tight_layout()
        
        elif chart_type == 'volcano':
            if 'cols' in config and len(config['cols']) >= 2:
                fc_col, pv_col = config['cols'][0], config['cols'][1]
            else:
                st.error("Volcano plot requires two numeric columns: a fold-change and a p-value column.")
                plt.close(fig)
                return None
            
            # Use log2 for fold change and -log10 for p-value
            df['log2_fold_change'] = np.log2(df[fc_col])
            # Add a small value to prevent log(0) errors
            df['-log10_p_value'] = -np.log10(df[pv_col] + 1e-300)

            # Define thresholds
            p_threshold = 0.05
            fc_threshold = 1.5
            
            # Create a significance column for coloring
            df['significance'] = 'Not significant'
            df.loc[(df['-log10_p_value'] > -np.log10(p_threshold)) & (df['log2_fold_change'] > np.log2(fc_threshold)), 'significance'] = 'Up-regulated'
            df.loc[(df['-log10_p_value'] > -np.log10(p_threshold)) & (df['log2_fold_change'] < -np.log2(fc_threshold)), 'significance'] = 'Down-regulated'

            # Plot the data
            sns.scatterplot(
                data=df,
                x='log2_fold_change',
                y='-log10_p_value',
                hue='significance',
                palette={'Up-regulated': 'red', 'Down-regulated': 'green', 'Not significant': 'gray'},
                ax=ax,
                s=20
            )

            # Add threshold lines
            ax.axhline(y=-np.log10(p_threshold), color='black', linestyle='--', linewidth=1, label=f'p-value={p_threshold}')
            ax.axvline(x=np.log2(fc_threshold), color='black', linestyle=':', linewidth=1, label=f'|Fold Change|={fc_threshold}')
            ax.axvline(x=-np.log2(fc_threshold), color='black', linestyle=':', linewidth=1)

            ax.set_title(title or 'Volcano Plot')
            ax.set_xlabel('$\log_2$ Fold Change' or x_label)
            ax.set_ylabel('$-\log_{10}$ P-value' or y_label)
            ax.legend(title='Regulation')
            plt.tight_layout()
        elif chart_type == 'tga':
            if 'cols' in config and len(config['cols']) >= 2:
                x_col, y_col = config['cols'][0], config['cols'][1]
            else:
                st.error("TGA curve requires two numeric columns: a temperature or time column and a mass column.")
                plt.close(fig)
                return None
            
            x_data = df[x_col]
            y_data = df[y_col]
            
            if pd.api.types.is_numeric_dtype(x_data) and pd.api.types.is_numeric_dtype(y_data):
                # Calculate mass percentage relative to the initial mass
                initial_mass = y_data.iloc[0]
                mass_percentage = (y_data / initial_mass) * 100
                
                ax.plot(x_data, mass_percentage, color=color, label='Mass Loss')
                
                ax.set_title(title or f'TGA Curve of {y_col} vs {x_col}')
                ax.set_xlabel(x_label or f'{x_col} (°C)' if 'temp' in x_col.lower() else x_col)
                ax.set_ylabel(y_label or 'Mass (%)')
                ax.legend()
            else:
                st.error("TGA curve requires both columns to be numeric.")
            plt.tight_layout()
        
        elif chart_type == 'roc':
            if 'cols' in config and len(config['cols']) >= 2:
                y_true_col, y_score_col = config['cols'][0], config['cols'][1]
            else:
                st.error("ROC curve requires two numeric columns: true labels and predicted scores/probabilities.")
                plt.close(fig)
                return None

            # Create a clean DataFrame for plotting
            plot_df = df[[y_true_col, y_score_col]].copy()
            
            # Drop rows with NaN or infinite values in the relevant columns
            plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            plot_df.dropna(inplace=True)

            if plot_df.empty:
                st.warning("The selected columns contain no valid data after cleaning. Please check your source file.")
                plt.close(fig)
                return None
            
            y_true = plot_df[y_true_col].astype(int)
            y_score = plot_df[y_score_col]

            if pd.api.types.is_numeric_dtype(y_true) and pd.api.types.is_numeric_dtype(y_score):
                try:
                    # Compute ROC curve and AUC
                    fpr, tpr, thresholds = roc_curve(y_true, y_score)
                    roc_auc = auc(fpr, tpr)

                    # Plot the ROC curve
                    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    
                    # Plot the random classifier line
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    
                    ax.set_title(title or 'Receiver Operating Characteristic (ROC) Curve')
                    ax.set_xlabel(x_label or 'False Positive Rate')
                    ax.set_ylabel(y_label or 'True Positive Rate')
                    ax.legend(loc='lower right')

                except ValueError as e:
                    st.error(f"Could not compute the ROC curve. Please ensure the data for '{y_true_col}' contains true binary labels (e.g., 0 or 1).")
                    plt.close(fig)
                    return None
            else:
                st.error("ROC curve requires both columns to be numeric.")
            plt.tight_layout()
        
        elif chart_type == 'logistic_regression':
            if 'cols' in config and len(config['cols']) >= 2:
                x_col, y_col = config['cols'][0], config['cols'][1]
            else:
                st.error("Logistic regression requires two numeric columns: an independent variable and a binary dependent variable.")
                plt.close(fig)
                return None
            
            x_data = df[x_col].values.reshape(-1, 1)
            y_data = df[y_col]

            if pd.api.types.is_numeric_dtype(x_data) and pd.api.types.is_numeric_dtype(y_data):
                try:
                    # Initialize and fit the logistic regression model
                    model = LogisticRegression(solver='liblinear')
                    model.fit(x_data, y_data)

                    # Create a scatter plot of the original data points
                    ax.scatter(x_data, y_data, color='blue', zorder=20, label='Data Points')

                    # Generate x-values for the smooth curve
                    x_test = np.linspace(x_data.min(), x_data.max(), 300).reshape(-1, 1)
                    
                    # Predict probabilities for the smooth curve
                    y_prob = model.predict_proba(x_test)[:, 1]

                    # Plot the logistic curve
                    ax.plot(x_test, y_prob, color='red', linewidth=3, label='Logistic Curve')

                    ax.set_title(title or f'Logistic Regression of {y_col} vs {x_col}')
                    ax.set_xlabel(x_label or x_col)
                    ax.set_ylabel(y_label or y_col)
                    ax.legend()
                    ax.grid(True)
                except ValueError as e:
                    st.error(f"Could not fit a logistic regression model. Please ensure the data for '{y_col}' contains only binary labels (0 or 1).")
                    plt.close(fig)
                    return None
            else:
                st.error("Logistic regression chart requires both columns to be numeric.")
            plt.tight_layout()
            
        elif chart_type == 'dtg':
            if 'cols' in config and len(config['cols']) >= 2:
                x_col, y_col = config['cols'][0], config['cols'][1]
            else:
                st.error("DTG curve requires two numeric columns: a temperature or time column and either a mass or pre-calculated dMass/dT column.")
                plt.close(fig)
                return None
            
            x_data = df[x_col]
            y_data = df[y_col]

            if pd.api.types.is_numeric_dtype(x_data) and pd.api.types.is_numeric_dtype(y_data):
                dtg_data_calculated = False
                # Check if the y-column is already a derivative (e.g., 'dMass/dT')
                if 'dtg' in y_col.lower() or 'dmass' in y_col.lower():
                    dtg_data = y_data
                    y_label_str = 'dMass/dT (%)'
                else:
                    # Calculate DTG if a mass column is provided
                    initial_mass = y_data.iloc[0]
                    mass_percentage = (y_data / initial_mass) * 100
                    # Use a negative sign to show mass loss as a positive peak
                    dtg_data = -np.gradient(mass_percentage, x_data)
                    y_label_str = 'dMass/dT (%)'
                    dtg_data_calculated = True
                    
                ax.plot(x_data, dtg_data, color=color, label='Mass Loss Rate')
                
                ax.set_title(title or f'DTG Curve of {x_col} vs dMass/dT')
                ax.set_xlabel(x_label or f'{x_col} (°C)' if 'temp' in x_col.lower() else x_col)
                ax.set_ylabel(y_label or y_label_str)
                ax.legend()
            else:
                st.error("DTG curve requires both columns to be numeric.")
            plt.tight_layout()
            
        elif chart_type == 'combo':
            if 'cols' in config and len(config['cols']) >= 3:
                x_col, bar_col, line_col = config['cols'][0], config['cols'][1], config['cols'][2]
            else:
                st.error("Combo chart requires three columns: one for the x-axis, one for the bars, and one for the line.")
                plt.close(fig)
                return None
            
            x_data = df[x_col]
            bar_data = df[bar_col]
            line_data = df[line_col]
            
            if pd.api.types.is_numeric_dtype(bar_data) and pd.api.types.is_numeric_dtype(line_data):
                # Create the first y-axis for the bars
                ax.bar(x_data, bar_data, color='tab:blue', label=bar_col)
                ax.set_ylabel(bar_col, color='tab:blue')
                ax.tick_params(axis='y', labelcolor='tab:blue')
                
                # Create the second y-axis for the line
                ax2 = ax.twinx()
                ax2.plot(x_data, line_data, color='tab:red', marker='o', label=line_col)
                ax2.set_ylabel(line_col, color='tab:red')
                ax2.tick_params(axis='y', labelcolor='tab:red')
                
                # Set common x-axis properties
                ax.set_xlabel(x_col)
                ax.set_title(title or f'Combo Chart of {bar_col} and {line_col} over {x_col}')

                # Combine legends from both axes
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper left')

            else:
                st.error("Combo chart requires the bar and line columns to be numeric.")
            
            plt.tight_layout()
            
        elif chart_type == 'smooth_curve':
            if 'cols' in config and len(config['cols']) >= 2:
                x_col, y_col = config['cols'][0], config['cols'][1]
            else:
                st.error("Smooth curve fit requires two numeric columns. Please specify them, e.g., 'show a smooth curve of time vs temperature'")
                plt.close(fig)
                return None
            
            # Create a clean DataFrame for plotting
            plot_df = df[[x_col, y_col]].copy()
            plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            plot_df.dropna(inplace=True)
            
            if plot_df.empty:
                st.warning("The selected columns contain no valid data after cleaning. Please check your source file.")
                plt.close(fig)
                return None
            
            x_data = plot_df[x_col]
            y_data = plot_df[y_col]
            
            if pd.api.types.is_numeric_dtype(x_data) and pd.api.types.is_numeric_dtype(y_data):
                try:
                    # Choose a lower polynomial degree for a more stable fit
                    degree = 2
                    
                    # Fit a polynomial to the data
                    coeffs = np.polyfit(x_data, y_data, degree)
                    poly_func = np.poly1d(coeffs)
                    
                    # Generate x-values for the smooth curve
                    x_fit = np.linspace(x_data.min(), x_data.max(), 500)
                    y_fit = poly_func(x_fit)
                    
                    ax.scatter(x_data, y_data, color='blue', label='Data Points')
                    ax.plot(x_fit, y_fit, color='red', label='Fitted Smooth Curve')
                    
                    ax.set_title(title or f'Smooth Curve Fit of {y_col} vs {x_col}')
                    ax.set_xlabel(x_label or x_col)
                    ax.set_ylabel(y_label or y_col)
                    ax.legend()
                
                except Exception as e:
                    st.error(f"Could not fit the smooth curve to the data. Please ensure the data is suitable for a polynomial fit. Error: {e}")
            else:
                st.error("Smooth curve fit requires both columns to be numeric.")
            plt.tight_layout()
            
        elif chart_type == 'ld_heatmap':
            if 'cols' in config and len(config['cols']) >= 2:
                # Get the columns (SNPs) for the heatmap
                ld_cols = config['cols']
            else:
                st.error("An LD heatmap requires at least two columns representing genetic markers.")
                plt.close(fig)
                return None
            
            # Check if all selected columns are numeric
            if all(pd.api.types.is_numeric_dtype(df[col]) for col in ld_cols):
                try:
                    # Calculate the correlation matrix (r^2 is just the square of the correlation)
                    ld_matrix = df[ld_cols].corr()**2

                    # Plot the heatmap
                    sns.heatmap(ld_matrix, annot=False, cmap='magma', ax=ax, cbar_kws={'label': 'Linkage Disequilibrium (r²)'})
                    
                    ax.set_title(title or 'Linkage Disequilibrium (LD) Heatmap')
                    
                except Exception as e:
                    st.error(f"Failed to generate the LD heatmap. Please ensure the selected columns are numeric. Error: {e}")
                    plt.close(fig)
                    return None
            else:
                st.error("LD heatmap requires all selected columns to be numeric.")
            plt.tight_layout()
            
        elif chart_type == 'tga_dtg_combo':
            if 'cols' in config and len(config['cols']) >= 2:
                x_col, y_col = config['cols'][0], config['cols'][1]
            else:
                st.error("TGA+DTG chart requires two numeric columns: a temperature or time column and a mass column.")
                plt.close(fig)
                return None
            
            x_data = df[x_col]
            y_data = df[y_col]
            
            if pd.api.types.is_numeric_dtype(x_data) and pd.api.types.is_numeric_dtype(y_data):
                # Calculate TGA mass percentage
                initial_mass = y_data.iloc[0]
                mass_percentage = (y_data / initial_mass) * 100
                
                # Create the first y-axis for the TGA curve
                ax.plot(x_data, mass_percentage, color='tab:blue', label='TGA (Mass %)')
                ax.set_ylabel('Mass (%)', color='tab:blue')
                ax.tick_params(axis='y', labelcolor='tab:blue')
                
                # Create the second y-axis for the DTG curve
                ax2 = ax.twinx()
                # Calculate DTG (negative derivative to show mass loss as a positive peak)
                dtg_data = -np.gradient(mass_percentage, x_data)
                ax2.plot(x_data, dtg_data, color='tab:red', label='DTG')
                ax2.set_ylabel('Rate of Mass Change (%/°C)', color='tab:red')
                ax2.tick_params(axis='y', labelcolor='tab:red')
                
                # Set common x-axis properties
                ax.set_title(title or 'Combined TGA-DTG Curve')
                ax.set_xlabel(x_label or f'{x_col} (°C)' if 'temp' in x_col.lower() else x_col)

                # Combine legends from both axes
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper right')

            else:
                st.error("Combined TGA-DTG chart requires both columns to be numeric.")
            
            plt.tight_layout()   

        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.6)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

    except Exception as e:
        st.error(f"Failed to generate chart: {e}")
        plt.close(fig)
        return None

# --- Main Application Logic ---
uploaded_files = st.file_uploader("Upload CSV, JSON, or XLSX files", type=['csv', 'json', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    try:
        combined_df = pd.DataFrame()
        
        # We'll use the first file's metadata for the report
        first_file = uploaded_files[0]
        st.session_state.uploaded_file_name = first_file.name
        st.session_state.uploaded_file_type = first_file.type
        
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Reset file pointer before reading
            uploaded_file.seek(0)

            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file, encoding='latin1')
            elif file_extension == '.json':
                df = pd.read_json(uploaded_file, encoding='latin1')
            elif file_extension == '.xlsx':
                df = pd.read_excel(uploaded_file, encoding='latin1')
            else:
                st.warning(f"Skipping unsupported file: {uploaded_file.name}")
                continue

            if df is not None:
                combined_df = pd.concat([combined_df, df], ignore_index=True)

        if not combined_df.empty:
            # Drop any columns that contain "Unnamed" in their name and are all null
            combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('^Unnamed') | combined_df.isnull().all()]
            
            st.session_state.combined_df = combined_df
            st.success("Files loaded successfully!")

            st.subheader("Data Preview")
            st.dataframe(st.session_state.combined_df.head())
            st.write(f"Total rows: {len(st.session_state.combined_df)}")
            st.write(f"Columns: {', '.join(st.session_state.combined_df.columns)}")

            st.markdown("---")
            
            # Wrap the analysis logic in a button click
            user_question = st.text_input("Ask a question or request a chart:", placeholder="e.g., What is the average height? or show a bar chart of gender.", value=st.session_state.user_question)
            st.session_state.user_question = user_question
            
            if st.button("Analyze"):
                if not user_question:
                    st.warning("Please enter a question to analyze the data.")
                elif not st.session_state.api_key:
                    st.error("Gemini API Key is not set. Please ensure it's in your .env file.")
                else:
                    question_lower = user_question.lower()
                    
                    # --- Chart Generation Logic ---
                    chart_request_found = False
                    chart_type = None
                    chart_config = {}

                    if "pie chart" in question_lower:
                        chart_type = 'pie'
                        for col in st.session_state.combined_df.columns:
                            if find_matching_column(st.session_state.combined_df, col) and (find_matching_column(st.session_state.combined_df, col).lower() in question_lower) and (st.session_state.combined_df[col].dtype == 'object' or st.session_state.combined_df[col].nunique() < 20):
                                chart_config = {'col': col}
                                chart_request_found = True
                                break

                    elif "bar chart" in question_lower or "bar graph" in question_lower:
                        chart_type = 'bar'
                        for col in st.session_state.combined_df.columns:
                            if find_matching_column(st.session_state.combined_df, col) and (find_matching_column(st.session_state.combined_df, col).lower() in question_lower) and (st.session_state.combined_df[col].dtype == 'object' or st.session_state.combined_df[col].nunique() < 50):
                                chart_config = {'col': col}
                                chart_request_found = True
                                break

                    elif "scatter plot" in question_lower:
                        chart_type = 'scatter'
                        # Look for two columns for scatter plot
                        match = re.search(r'scatter plot of (.+) vs (.+)', question_lower)
                        if match:
                            x_query, y_query = match.groups()
                            x_col = find_matching_column(combined_df, x_query)
                            y_col = find_matching_column(combined_df, y_query)
                            if x_col and y_col and pd.api.types.is_numeric_dtype(combined_df[x_col]) and pd.api.types.is_numeric_dtype(combined_df[y_col]):
                                chart_config = {'cols': [x_col, y_col]}
                                chart_request_found = True

                    elif "violin plot" in question_lower:
                        chart_type = 'violin'
                        chart_config = {} # We'll handle this generically in the function
                        chart_request_found = True

                    elif "box plot" in question_lower:
                        chart_type = 'box'
                        # Box plots can be of all numeric columns or a specific one
                        match = re.search(r'box plot of (.+)', question_lower)
                        if match:
                            col_query = match.group(1)
                            col = find_matching_column(combined_df, col_query)
                            if col and pd.api.types.is_numeric_dtype(combined_df[col]):
                                chart_config = {'cols': [col]}
                                chart_request_found = True
                        else:
                            # Default to all numeric columns if not specified
                            numeric_cols = combined_df.select_dtypes(include=np.number).columns.tolist()
                            if numeric_cols:
                                chart_config = {'cols': numeric_cols}
                                chart_request_found = True
                        
                    elif "heatmap" in question_lower:
                        chart_type = 'heatmap'
                        # If two columns are specified, create a contingency table heatmap
                        match = re.search(r'heatmap of (.+) vs (.+)', question_lower)
                        if match:
                            x_query, y_query = match.groups()
                            x_col = find_matching_column(combined_df, x_query)
                            y_col = find_matching_column(combined_df, y_query)
                            if x_col and y_col:
                                chart_config = {'cols': [x_col, y_col]}
                                chart_request_found = True
                        else:
                            # Default to all numeric columns if not specified
                            numeric_cols = combined_df.select_dtypes(include=np.number).columns.tolist()
                            if numeric_cols:
                                chart_config = {'cols': numeric_cols}
                                chart_request_found = True
                    
                    elif "exponential curve" in question_lower or "exponential plot" in question_lower:
                        chart_type = 'exponential'
                        parts = question_lower.split(" of ")
                        if len(parts) == 2:
                            col_parts = parts[1].split(" vs ")
                            if len(col_parts) == 2:
                                x_query = col_parts[0].strip()
                                y_query = col_parts[1].strip()
                                x_col = find_matching_column(combined_df, x_query)
                                y_col = find_matching_column(combined_df, y_query)
                                if x_col and y_col and pd.api.types.is_numeric_dtype(combined_df[x_col]) and pd.api.types.is_numeric_dtype(combined_df[y_col]):
                                    chart_config = {'cols': [x_col, y_col]}
                                    chart_request_found = True       
                    # --- START of NEW LOGIC for LINE CHART ---
                    elif "line chart" in question_lower:
                        chart_type = 'line'
                        
                        # First, try to find a single column
                        for col in st.session_state.combined_df.columns:
                            if find_matching_column(st.session_state.combined_df, col) and (find_matching_column(st.session_state.combined_df, col).lower() in question_lower):
                                if pd.api.types.is_numeric_dtype(st.session_state.combined_df[col]):
                                    chart_config = {'col': col}
                                    chart_request_found = True
                                    break
                        
                        # If a single column wasn't found, try the original two-column regex
                        if not chart_request_found:
                            match = re.search(r'line chart of (.+) vs (.+)', question_lower) or re.search(r'line chart of (.+) over (.+)', question_lower)
                            if match:
                                x_query, y_query = match.groups()
                                x_col = find_matching_column(combined_df, x_query)
                                y_col = find_matching_column(combined_df, y_query)
                                if x_col and y_col and (pd.api.types.is_numeric_dtype(combined_df[x_col]) or pd.api.types.is_datetime64_any_dtype(combined_df[x_col])) and pd.api.types.is_numeric_dtype(combined_df[y_col]):
                                    chart_config = {'cols': [x_col, y_col]}
                                    chart_request_found = True
                    # --- END of NEW LOGIC for LINE CHART ---
                    elif "logarithmic curve" in question_lower or "logarithmic plot" in question_lower:
                        chart_type = 'logarithmic'
                        parts = question_lower.split(" of ")
                        if len(parts) == 2:
                            col_parts = parts[1].split(" vs ")
                            if len(col_parts) == 2:
                                x_query = col_parts[0].strip()
                                y_query = col_parts[1].strip()
                                x_col = find_matching_column(combined_df, x_query)
                                y_col = find_matching_column(combined_df, y_query)
                                if x_col and y_col and pd.api.types.is_numeric_dtype(combined_df[x_col]) and pd.api.types.is_numeric_dtype(combined_df[y_col]):
                                    chart_config = {'cols': [x_col, y_col]}
                                    chart_request_found = True
                    
                    elif "volcano plot" in question_lower or "volcano chart" in question_lower:
                        chart_type = 'volcano'
                        parts = question_lower.split(" of ")
                        if len(parts) == 2:
                            col_parts = parts[1].split(" vs ")
                            if len(col_parts) == 2:
                                fc_query = col_parts[0].strip()
                                pv_query = col_parts[1].strip()
                                fc_col = find_matching_column(combined_df, fc_query)
                                pv_col = find_matching_column(combined_df, pv_query)
                                if fc_col and pv_col and pd.api.types.is_numeric_dtype(combined_df[fc_col]) and pd.api.types.is_numeric_dtype(combined_df[pv_col]):
                                    chart_config = {'cols': [fc_col, pv_col]}
                                    chart_request_found = True            
                    elif "tga curve" in question_lower or "thermogravimetric analysis" in question_lower:
                        chart_type = 'tga'
                        parts = question_lower.split(" of ")
                        if len(parts) == 2:
                            col_parts = parts[1].split(" vs ")
                            if len(col_parts) == 2:
                                x_query = col_parts[0].strip()
                                y_query = col_parts[1].strip()
                                x_col = find_matching_column(combined_df, x_query)
                                y_col = find_matching_column(combined_df, y_query)
                                if x_col and y_col and pd.api.types.is_numeric_dtype(combined_df[x_col]) and pd.api.types.is_numeric_dtype(combined_df[y_col]):
                                    chart_config = {'cols': [x_col, y_col]}
                                    chart_request_found = True
                    elif "tga curve" in question_lower or "thermogravimetric analysis" in question_lower:
                        chart_type = 'tga'
                        parts = question_lower.split(" of ")
                        if len(parts) == 2:
                            col_parts = parts[1].split(" vs ")
                            if len(col_parts) == 2:
                                x_query = col_parts[0].strip()
                                y_query = col_parts[1].strip()
                                x_col = find_matching_column(combined_df, x_query)
                                y_col = find_matching_column(combined_df, y_query)
                                if x_col and y_col and pd.api.types.is_numeric_dtype(combined_df[x_col]) and pd.api.types.is_numeric_dtype(combined_df[y_col]):
                                    chart_config = {'cols': [x_col, y_col]}
                                    chart_request_found = True
                
                    elif "roc curve" in question_lower or "roc plot" in question_lower:
                        chart_type = 'roc'
                        parts = question_lower.split(" of ")
                        if len(parts) == 2:
                            col_parts = parts[1].split(" vs ")
                            if len(col_parts) == 2:
                                y_true_query = col_parts[0].strip()
                                y_score_query = col_parts[1].strip()
                                y_true_col = find_matching_column(combined_df, y_true_query)
                                y_score_col = find_matching_column(combined_df, y_score_query)
                                if y_true_col and y_score_col and pd.api.types.is_numeric_dtype(combined_df[y_true_col]) and pd.api.types.is_numeric_dtype(combined_df[y_score_col]):
                                    chart_config = {'cols': [y_true_col, y_score_col]}
                                    chart_request_found = True
                
                    elif "logistic regression" in question_lower or "logistic curve" in question_lower:
                        chart_type = 'logistic_regression'
                        parts = question_lower.split(" of ")
                        if len(parts) == 2:
                            col_parts = parts[1].split(" vs ")
                            if len(col_parts) == 2:
                                x_query = col_parts[0].strip()
                                y_query = col_parts[1].strip()
                                x_col = find_matching_column(combined_df, x_query)
                                y_col = find_matching_column(combined_df, y_query)
                                if x_col and y_col and pd.api.types.is_numeric_dtype(combined_df[x_col]) and pd.api.types.is_numeric_dtype(combined_df[y_col]):
                                    chart_config = {'cols': [x_col, y_col]}
                                    chart_request_found = True
                    elif "dtg curve" in question_lower or "derivative thermogravimetry" in question_lower:
                        chart_type = 'dtg'
                        parts = question_lower.split(" of ")
                        if len(parts) == 2:
                            col_parts = parts[1].split(" vs ")
                            if len(col_parts) == 2:
                                x_query = col_parts[0].strip()
                                y_query = col_parts[1].strip()
                                x_col = find_matching_column(combined_df, x_query)
                                y_col = find_matching_column(combined_df, y_query)
                                if x_col and y_col and pd.api.types.is_numeric_dtype(combined_df[x_col]) and pd.api.types.is_numeric_dtype(combined_df[y_col]):
                                    chart_config = {'cols': [x_col, y_col]}
                                    chart_request_found = True
                                
                    elif "combo chart" in question_lower or "line and bar chart" in question_lower:
                        chart_type = 'combo'
                        parts = question_lower.split(" of ")
                        if len(parts) == 2:
                            col_parts = parts[1].split(", and ")
                            if len(col_parts) == 2:
                                x_query = col_parts[0].split(",")[0].strip()
                                bar_query = col_parts[0].split(",")[1].strip()
                                line_query = col_parts[1].strip()
                                x_col = find_matching_column(combined_df, x_query)
                                bar_col = find_matching_column(combined_df, bar_query)
                                line_col = find_matching_column(combined_df, line_query)
                                if x_col and bar_col and line_col and pd.api.types.is_numeric_dtype(combined_df[bar_col]) and pd.api.types.is_numeric_dtype(combined_df[line_col]):
                                    chart_config = {'cols': [x_col, bar_col, line_col]}
                                    chart_request_found = True
                
                    elif "smooth curve" in question_lower or "polynomial curve" in question_lower:
                        chart_type = 'smooth_curve'
                        parts = question_lower.split(" of ")
                        if len(parts) == 2:
                            col_parts = parts[1].split(" vs ")
                            if len(col_parts) == 2:
                                x_query = col_parts[0].strip()
                                y_query = col_parts[1].strip()
                                x_col = find_matching_column(combined_df, x_query)
                                y_col = find_matching_column(combined_df, y_query)
                                if x_col and y_col and pd.api.types.is_numeric_dtype(combined_df[x_col]) and pd.api.types.is_numeric_dtype(combined_df[y_col]):
                                    chart_config = {'cols': [x_col, y_col]}
                                    chart_request_found = True
                
                    elif "ld heatmap" in question_lower or "linkage disequilibrium" in question_lower:
                        chart_type = 'ld_heatmap'
                        parts = question_lower.split(" of ")
                        if len(parts) == 2:
                            col_parts = parts[1].split(", ")
                            ld_cols = [find_matching_column(combined_df, query.strip()) for query in col_parts]
                        
                            if all(ld_cols):
                                chart_config = {'cols': [col for col in ld_cols if col is not None]}
                                chart_request_found = True
                            
                    elif "tga+dtg" in question_lower or "combined thermal analysis" in question_lower:
                        chart_type = 'tga_dtg_combo'
                        parts = question_lower.split(" of ")
                        if len(parts) == 2:
                            col_parts = parts[1].split(" vs ")
                            if len(col_parts) == 2:
                                x_query = col_parts[0].strip()
                                y_query = col_parts[1].strip()
                                x_col = find_matching_column(combined_df, x_query)
                                y_col = find_matching_column(combined_df, y_query)
                                if x_col and y_col and pd.api.types.is_numeric_dtype(combined_df[x_col]) and pd.api.types.is_numeric_dtype(combined_df[y_col]):
                                    chart_config = {'cols': [x_col, y_col]}
                                    chart_request_found = True 
                    
                    if chart_request_found:
                        if not chart_config:
                            st.warning(f"Could not find a valid column for a {chart_type}. Please try again and specify the column name clearly. \n\n**Example:** `Show a {chart_type} of [Column Name]`")
                        else:
                            st.subheader("Chart Analysis")
                            with st.expander("Chart Options", expanded=False):
                                col_name = chart_config.get('col', "")
                                x_label_input = st.text_input("X-Axis Label", value=col_name)
                                y_label_input = st.text_input("Y-Axis Label", value='Count')
                                title_input = st.text_input("Chart Title", value="")
                                color_picker = st.color_picker("Chart Color", "#1f77b4")
                                show_grid = st.checkbox("Show grid", value=False)
                            
                                chart_config['x_label'] = x_label_input
                                chart_config['y_label'] = y_label_input
                                chart_config['title'] = title_input
                                chart_config['show_grid'] = show_grid
                                chart_config['color'] = color_picker

                            with st.spinner("Generating chart..."):
                                chart_bytes = generate_chart_bytes(st.session_state.combined_df, chart_type, chart_config)
                                st.session_state.chart_bytes = chart_bytes # Store bytes in session state

                            if st.session_state.chart_bytes:
                                st.image(st.session_state.chart_bytes, caption=f"Generated {chart_type} chart", use_column_width=True)
                                st.download_button(
                                    label=f"Download {chart_type.capitalize()} Chart as PNG",
                                    data=st.session_state.chart_bytes,
                                    file_name=f"{chart_type}_chart.png",
                                    mime="image/png"
                                )
                            else:
                                st.warning("Could not generate the chart. Please check the column names and data types.")

                    else:
                        # --- Mathematical Operations ---
                        math_operation = None
                        math_col = None

                        math_keywords = {'average': np.mean, 'mean': np.mean, 'sum': np.sum, 'total': np.sum, 'min': np.min, 'max': np.max}

                        for keyword, operation in math_keywords.items():
                            if keyword in question_lower:
                                math_operation = operation
                                for col in st.session_state.combined_df.columns:
                                    if find_matching_column(st.session_state.combined_df, col) and find_matching_column(st.session_state.combined_df, col).lower() in question_lower and pd.api.types.is_numeric_dtype(st.session_state.combined_df[col]):
                                        math_col = col
                                        break
                                break

                        if math_operation and math_col:
                            st.subheader("Mathematical Calculation")
                            try:
                                result = math_operation(st.session_state.combined_df[math_col])
                                if math_operation.__name__ == 'mean':
                                    st.info(f"The **average** of the `{math_col}` column is **{result:.2f}**.")
                                elif math_operation.__name__ == 'sum':
                                    st.info(f"The **total** of the `{math_col}` column is **{result:.2f}**.")
                                elif math_operation.__name__ == 'amin':
                                    st.info(f"The **minimum** value in the `{math_col}` column is **{result:.2f}**.")
                                elif math_operation.__name__ == 'amax':
                                    st.info(f"The **maximum** value in the `{math_col}` column is **{result:.2f}**.")
                            except Exception as e:
                                st.error(f"Could not perform calculation on column '{math_col}': {e}")
                        else:
                            st.subheader("AI Analyst Response")
                            with st.spinner("Getting a response from Gemini..."):
                                df_info = {
                                    'head': st.session_state.combined_df.head().to_string(),
                                    'rows': len(st.session_state.combined_df),
                                    'dtypes': st.session_state.combined_df.dtypes.to_string()
                                }
                                # Corrected function call to use st.session_state.api_key
                                api_response = get_answer_from_gemini(st.session_state.api_key, df_info, user_question)
                            st.markdown(api_response)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# ---
st.markdown("### Generate Final Report")
st.markdown("Use the options below to create a full report based on your data and analysis.")

with st.form("report_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        report_format = st.selectbox(
            "Report Format",
            ("docx", "pdf"),
            help="Choose the desired format for the final report."
        )
    with col2:
        model_id = st.selectbox(
            "AI Model",
            ("gemini-1.5-flash", "gemini-1.5-pro"),
            help="Select the AI model to be used for content generation."
        )
    
    submit_button = st.form_submit_button("Generate Report")
    
    # Submission Logic
    if submit_button:
        if st.session_state.combined_df is None:
            st.error("Please upload a file before generating the report.")
            st.stop()
        
        try:
            # Prepare the data to be sent to FastAPI backend
            data_bytes = io.BytesIO()
            if st.session_state.uploaded_file_type == 'text/csv':
                st.session_state.combined_df.to_csv(data_bytes, index=False)
                mime_type = 'text/csv'
            elif st.session_state.uploaded_file_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
                st.session_state.combined_df.to_excel(data_bytes, index=False)
                mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            data_bytes.seek(0)
            
            # Prepare the files dictionary
            files = {
                'file': (st.session_state.uploaded_file_name, data_bytes, mime_type)
            }
            
            # Create a dictionary for other form data
            data = {
                'user_query': st.session_state.user_question,
                'report_format': report_format,
                'model_id': model_id,
            }

            # Add chart image if it exists in session state
            if st.session_state.chart_bytes:
                chart_file = io.BytesIO(st.session_state.chart_bytes)
                files['chart_image'] = ('chart.png', chart_file, 'image/png')
            
            with st.spinner("Generating final report... This may take a moment."):
                response = requests.post(f"{FASTAPI_BACKEND_URL}/generate_report", data=data, files=files)
                response.raise_for_status()

            # Store the report content and metadata in session state
            st.session_state.report_content = response.content
            st.session_state.report_mime = response.headers['Content-Type']
            
            content_disposition = response.headers.get('Content-Disposition')
            filename = 'generated_report.docx'
            if content_disposition:
                filename_match = content_disposition.split('filename=')[-1].strip('"')
                filename = filename_match
            st.session_state.report_filename = filename

            st.success("Report generated successfully!")

        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while connecting to the backend: {e}. Please ensure your backend is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Display the download button only when the report content is ready
if st.session_state.report_content and st.session_state.report_filename and st.session_state.report_mime:
    st.download_button(
        label="Download Report",
        data=st.session_state.report_content,
        file_name=st.session_state.report_filename,
        mime=st.session_state.report_mime,
        help="Click to download your generated report."
    )

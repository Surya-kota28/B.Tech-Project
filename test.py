import streamlit as st
import requests as re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import machine_learning as ml
import feature_extraction as fe
import warnings
import time
from urllib.parse import urlparse

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Function to plot the pie chart
def plot_pie_chart(phishing_rate, legitimate_rate):
    labels = 'phishing', 'legitimate'
    sizes = [phishing_rate, legitimate_rate]
    explode = (0.1, 0)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    ax.axis('equal')
    return fig

def check_urlscan_api(api_key, url):
    urlscan_submit_url = "https://urlscan.io/api/v1/scan/"
    result_base_url = "https://urlscan.io/api/v1/result/"
    headers = {
        "API-Key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "url": url,
        "visibility": "public"  # Public scans
    }
    
    try:
        # Submit the URL for scanning
        response = re.post(urlscan_submit_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Extract scan ID from the response
        scan_id = data.get("uuid")
        if not scan_id:
            st.error("No scan ID received. Please try again.")
            return

        st.write(f"Scan submitted successfully. Scan ID: {scan_id}")
        st.write("Waiting for the scan to complete...")

        # Wait and retry fetching results
        max_attempts = 5
        for attempt in range(max_attempts):
            time.sleep(10)  # Delay between attempts
            result_url = f"{result_base_url}{scan_id}/"
            result_response = re.get(result_url)
            
            if result_response.status_code == 200:
                # Process the scan results
                scan_result = result_response.json()
                verdicts = scan_result.get('verdicts', {}).get('overall', {})
                categories = verdicts.get('categories', [])
                malicious = verdicts.get('malicious', False)
                score = verdicts.get('score', 0)

                st.subheader("Scan Results")
                st.write(f"URL: {url}")
                st.write(f"Malicious: {malicious}")
                
                if categories:
                    st.write(f"Categories: {', '.join(categories)}")
                else:
                    st.write("No categories associated with this URL.")

                # Provide the link to the full scan details
                st.write(f"[View full scan details here]({data.get('result')})")
                return
            elif result_response.status_code == 404:
                st.write(f"Attempt {attempt + 1}/{max_attempts}: Scan not ready yet.")
            else:
                result_response.raise_for_status()

        st.error("Scan results could not be retrieved after multiple attempts. Please check manually.")
        
    except re.exceptions.HTTPError as e:
        st.error(f"HTTP error occurred: {e}")
    except re.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")

# Function to check URL using the ML model
def check_url(url, model):
    if not url.startswith('http'):
        url = 'http://' + url

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = re.get(url, headers=headers, verify=True, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            st.write(f"Connection successful. Response code: {response.status_code}")

            soup = BeautifulSoup(response.content, "html.parser")
            vector = [fe.create_vector(soup)]  # Create feature vector for prediction (2D array)

            result = model.predict(vector)

            if result[0] == 0:
                st.success("This web page seems legitimate!")
                st.balloons()
            else:
                st.warning("Attention! This web page is a potential phishing attempt!")
                st.snow()
        else:
            st.write(f"HTTP connection was not successful. Response code: {response.status_code}")

    except re.exceptions.SSLError as e:
        st.error(f"SSL error occurred: {e}")
    except re.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")

# Streamlit app layout
st.title('Phishing Website Classification')

# Project details expander
with st.expander("PROJECT DETAILS"):
    st.subheader('Approach')
    st.write('This application uses supervised learning to classify phishing and legitimate websites using two methods: a content-based machine learning model and URL scanning using Urlscan.io API.')
    st.subheader('Data set')
    st.write('I used _"phishtank.org"_ & _"tranco-list.eu"_ as data sources.')
    st.write('Totally 26584 websites ==> **_16060_ legitimate** websites | **_10524_ phishing** websites')
    st.write('Data set was created in October 2022.')

    # Pie chart
    phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
    legitimate_rate = 100 - phishing_rate
    fig = plot_pie_chart(phishing_rate, legitimate_rate)
    st.pyplot(fig)

    st.write('Features + URL + Label ==> Dataframe')
    st.markdown('label is 1 for phishing, 0 for legitimate')
    number = st.slider("Select row number to display", 0, 100)
    st.dataframe(ml.legitimate_df.head(number))

# Model selection
choice = st.selectbox("Please select your method",
                      ['Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 
                       'Random Forest', 'AdaBoost', 'Neural Network', 'K-Neighbors', 
                       'Check URL with Urlscan.io'])

# Model dictionary
model_dict = {
    'Gaussian Naive Bayes': ml.nb_model,
    'Support Vector Machine': ml.svm_model,
    'Decision Tree': ml.dt_model,
    'Random Forest': ml.rf_model,
    'AdaBoost': ml.ab_model,
    'Neural Network': ml.nn_model,
    'K-Neighbors': ml.kn_model
}

# URL input
url = st.text_input('Enter the URL')

# URLScan API key input
api_key = st.text_input("Enter your Urlscan API key", type="password")

if choice == 'Check URL with Urlscan.io':
    if st.button('Check URL on Urlscan.io'):
        if api_key and url:
            check_urlscan_api(api_key, url)
        else:
            st.error('Please enter a valid Urlscan.io API key and URL.')
else:
    model = model_dict.get(choice, ml.nb_model)
    st.write(f'{choice} model is selected!')

    if st.button('Check with ML Model'):
        if url:
            check_url(url, model)
        else:
            st.error("Please enter a URL.")

import feature_extraction as fe

def display_feature_extraction_results(url):
    if not url.startswith('http'):
        url = 'http://' + url

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = re.get(url, headers=headers, verify=True, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            soup = fe.create_soup(response.content)
            vector = fe.create_vector(soup)
            feature_names = [
                'has_title',
                'has_input',
                'has_button',
                'has_image',
                'has_submit',
                'has_link',
                'has_password',
                'has_email_input',
                'has_hidden_element',
                'has_audio',
                'has_video',
                'number_of_inputs',
                'number_of_buttons',
                'number_of_images',
                'number_of_option',
                'number_of_list',
                'number_of_th',
                'number_of_tr',
                'number_of_href',
                'number_of_paragraph',
                'number_of_script',
                'length_of_title',
                'has_h1',
                'has_h2',
                'has_h3',
                'length_of_text',
                'number_of_clickable_button',
                'number_of_a',
                'number_of_img',
                'number_of_div',
                'number_of_figure',
                'has_footer',
                'has_form',
                'has_text_area',
                'has_iframe',
                'has_text_input',
                'number_of_meta',
                'has_nav',
                'has_object',
                'has_picture',
                'number_of_sources',
                'number_of_span',
                'number_of_table'
            ]
            feature_dict = dict(zip(feature_names, vector))
            
            # Display features as a table in Streamlit
            st.subheader("Extracted Features")
            st.table(feature_dict.items())
        else:
            st.error(f"HTTP connection was not successful. Response code: {response.status_code}")
    except re.exceptions.SSLError as e:
        st.error(f"SSL error occurred: {e}")
    except re.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")

if st.button('Display Feature Extraction Results'):
    if url:
        display_feature_extraction_results(url)
    else:
        st.error("Please enter a URL.")

import base64

def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the background
set_bg_from_local("calm-bg1.jpg")

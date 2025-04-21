import streamlit as st
import requests as re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import machine_learning as ml
import feature_extraction as fe
import warnings

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

# Function to check URL
def check_url(url, model):
    try:
        response = re.get(url, verify=False, timeout=4)
        if response.status_code != 200:
            st.write("HTTP connection was not successful.")
            return

        soup = BeautifulSoup(response.content, "html.parser")
        vector = [fe.create_vector(soup)]  # it should be 2d array, so I added []
        result = model.predict(vector)
        if result[0] == 0:
            st.success("This web page seems legitimate!")
            st.balloons()
        else:
            st.warning("Attention! This web page is a potential phishing attempt!")
            st.snow()

    except re.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")

# Streamlit app layout
st.title('Phishing Website Detection using Machine Learning')
st.write('This ML-based app is developed for educational purposes. The objective of the app is detecting phishing websites only using content data. Not URL! You can see the details of the approach, data set, and feature set if you click on _"See The Details"._ ')

with st.expander("PROJECT DETAILS"):
    st.subheader('Approach')
    st.write('I used _supervised learning_ to classify phishing and legitimate websites. I benefit from a content-based approach and focus on HTML of the websites. Also, I used scikit-learn for the ML models.')
    st.write('For this educational project, I created my own data set and defined features, some from the literature and some based on manual analysis. I used the requests library to collect data and BeautifulSoup module to parse and extract features.')
    st.write('The source code and data sets are available in the below Github link:')
    st.write('_https://github.com/emre-kocyigit/phishing-website-detection-content-based_')

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

    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(ml.df)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='phishing_legitimate_structured_data.csv',
        mime='text/csv',
    )

    st.subheader('Features')
    st.write('I used only content-based features. I didn\'t use URL-based features like length of URL, etc. Most of the features were extracted using the find_all() method of the BeautifulSoup module after parsing HTML.')

    st.subheader('Results')
    st.write('I used 7 different ML classifiers of scikit-learn and tested them implementing k-fold cross-validation. Firstly obtained their confusion matrices, then calculated their accuracy, precision, and recall scores. Comparison table is below:')
    st.table(ml.df_results)
    st.write('NB --> Gaussian Naive Bayes')
    st.write('SVM --> Support Vector Machine')
    st.write('DT --> Decision Tree')
    st.write('RF --> Random Forest')
    st.write('AB --> AdaBoost')
    st.write('NN --> Neural Network')
    st.write('KN --> K-Neighbors')

with st.expander('EXAMPLE PHISHING URLs:'):
    st.write('_https://rtyu38.godaddysites.com/_')
    st.write('_https://karafuru.invite-mint.com/_')
    st.write('_https://defi-ned.top/h5/#/_')
    st.caption('REMEMBER, PHISHING WEB PAGES HAVE SHORT LIFECYCLE! SO, THE EXAMPLES SHOULD BE UPDATED!')

choice = st.selectbox("Please select your machine learning model",
                 [
                     'Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
                     'AdaBoost', 'Neural Network', 'K-Neighbors'
                 ]
                )

# Model selection
model_dict = {
    'Gaussian Naive Bayes': ml.nb_model,
    'Support Vector Machine': ml.svm_model,
    'Decision Tree': ml.dt_model,
    'Random Forest': ml.rf_model,
    'AdaBoost': ml.ab_model,
    'Neural Network': ml.nn_model,
    'K-Neighbors': ml.kn_model
}

model = model_dict.get(choice, ml.nb_model)
st.write(f'{choice} model is selected!')

# URL input and checking
url = st.text_input('Enter the URL')
if st.button('Check!'):
    check_url(url, model)

import streamlit as st
import requests as re
from bs4 import BeautifulSoup
import feature_extraction as fe
import pandas as pd
import warnings
import time

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Phishing Website Detector", page_icon="🛡️", layout="centered")

PAPER_URL = "https://ieeexplore.ieee.org/document/10967695"
GITHUB_URL = "https://github.com/Surya-kota28"
PORTFOLIO_URL = "https://surya-kota28.github.io/Jasper-Surya"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

FEATURE_NAMES = [
    'has_title', 'has_input', 'has_button', 'has_image', 'has_submit', 'has_link',
    'has_password', 'has_email_input', 'has_hidden_element', 'has_audio', 'has_video',
    'number_of_inputs', 'number_of_buttons', 'number_of_images', 'number_of_option',
    'number_of_list', 'number_of_th', 'number_of_tr', 'number_of_href',
    'number_of_paragraph', 'number_of_script', 'length_of_title', 'has_h1', 'has_h2',
    'has_h3', 'length_of_text', 'number_of_clickable_button', 'number_of_a',
    'number_of_img', 'number_of_div', 'number_of_figure', 'has_footer', 'has_form',
    'has_text_area', 'has_iframe', 'has_text_input', 'number_of_meta', 'has_nav',
    'has_object', 'has_picture', 'number_of_sources', 'number_of_span', 'number_of_table'
]

URLSCAN_OPTION = 'Urlscan.io (external scan)'

st.markdown("""
<style>
.verdict {padding: 1.2rem 1.4rem; border-radius: 10px; margin: 0.5rem 0 1rem;}
.verdict h2 {margin: 0; padding: 0; font-size: 1.8rem;}
.verdict p {margin: 0.3rem 0 0; opacity: 0.85;}
.verdict.safe {background: rgba(34,197,94,0.12); border-left: 6px solid #22C55E;}
.verdict.safe h2 {color: #4ADE80;}
.verdict.danger {background: rgba(239,68,68,0.12); border-left: 6px solid #EF4444;}
.verdict.danger h2 {color: #F87171;}
.footer {text-align: center; opacity: 0.6; font-size: 0.85rem; margin-top: 3rem;}
.footer a {color: inherit;}
button[kind^="primary"], button[kind^="primary"] p {color: #0B1426 !important; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


def normalize_url(url):
    url = url.strip()
    if not url.startswith('http'):
        url = 'http://' + url
    return url


# Fetch the page and build the feature vector once, so both tabs use the same scan
def scan_with_model(url, model_name):
    url = normalize_url(url)
    try:
        response = re.get(url, headers=HEADERS, verify=True, timeout=10, allow_redirects=True)
    except re.exceptions.SSLError as e:
        return {"error": f"SSL error occurred: {e}"}
    except re.exceptions.RequestException as e:
        return {"error": f"Could not reach the page: {e}"}

    if response.status_code != 200:
        return {"error": f"HTTP connection was not successful. Response code: {response.status_code}"}

    soup = BeautifulSoup(response.content, "html.parser")
    vector = fe.create_vector(soup)
    model = MODEL_DICT[model_name]
    features = pd.DataFrame([vector], columns=ml.X.columns)

    label = int(model.predict(features)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(features)[0][label])

    return {"url": url, "model": model_name, "label": label,
            "confidence": confidence, "vector": vector}


def render_verdict(result):
    if result["label"] == 0:
        css, title, text = "safe", "Legitimate", "This page does not show the patterns of a phishing site."
    else:
        css, title, text = "danger", "Phishing", "This page looks like a potential phishing attempt. Do not enter credentials."

    st.markdown(f'<div class="verdict {css}"><h2>{title}</h2><p>{text}</p></div>',
                unsafe_allow_html=True)

    cols = st.columns(2)
    cols[0].metric("Confidence", f'{result["confidence"]:.0%}' if result["confidence"] is not None else "n/a")
    cols[1].metric("Features analysed", len(result["vector"]))
    st.caption(f'Scanned with {result["model"]}: {result["url"]}')


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

        scan_id = data.get("uuid")
        if not scan_id:
            st.error("No scan ID received. Please try again.")
            return

        with st.status(f"Scan submitted (ID: {scan_id}). Waiting for results...") as status:
            max_attempts = 5
            for attempt in range(max_attempts):
                time.sleep(10)  # Delay between attempts
                result_response = re.get(f"{result_base_url}{scan_id}/")

                if result_response.status_code == 200:
                    status.update(label="Scan complete", state="complete")
                    verdicts = result_response.json().get('verdicts', {}).get('overall', {})
                    malicious = verdicts.get('malicious', False)
                    categories = verdicts.get('categories', [])
                    break
                elif result_response.status_code == 404:
                    st.write(f"Attempt {attempt + 1}/{max_attempts}: scan not ready yet.")
                else:
                    result_response.raise_for_status()
            else:
                status.update(label="Scan timed out", state="error")
                st.error("Scan results could not be retrieved after multiple attempts. Please check manually.")
                return

        if malicious:
            st.markdown('<div class="verdict danger"><h2>Malicious</h2><p>Urlscan.io flagged this URL.</p></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict safe"><h2>Not flagged</h2><p>Urlscan.io did not flag this URL as malicious.</p></div>',
                        unsafe_allow_html=True)
        st.write(f"Categories: {', '.join(categories) if categories else 'none'}")
        st.markdown(f"[View full scan details on Urlscan.io]({data.get('result')})")

    except re.exceptions.HTTPError as e:
        st.error(f"HTTP error occurred: {e}")
    except re.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")


# Header
st.title("Phishing Website Detector")
st.markdown(f"Machine-learning phishing detection with an optional Urlscan.io check · "
            f"[IEEE ICMLAS 2025]({PAPER_URL})")


@st.cache_resource(show_spinner=False)
def load_models():
    # Importing the module trains the models; cached so it only happens once per server start
    import machine_learning
    return machine_learning


with st.spinner("Loading the detection models. This takes a few seconds on first visit..."):
    ml = load_models()

MODEL_DICT = {
    'Random Forest': ml.rf_model,
    'AdaBoost': ml.ab_model,
    'Decision Tree': ml.dt_model,
    'Neural Network': ml.nn_model,
    'K-Neighbors': ml.kn_model,
    'Support Vector Machine': ml.svm_model,
    'Gaussian Naive Bayes': ml.nb_model,
}

# Sidebar: settings
with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Detection method", list(MODEL_DICT) + [URLSCAN_OPTION])
    api_key = ""
    if choice == URLSCAN_OPTION:
        api_key = st.text_input("Urlscan.io API key", type="password",
                                help="Free key from urlscan.io. Used only for this request.")
        st.caption("This option needs your own API key. Create a free account at "
                   "[urlscan.io](https://urlscan.io/user/signup), then copy your key from your profile settings.")
    st.divider()
    st.caption("ML models are trained on HTML content features extracted from each page.")

n_legit = ml.legitimate_df.shape[0]
n_phish = ml.phishing_df.shape[0]
stats = st.columns(3)
stats[0].metric("Websites in dataset", f"{n_legit + n_phish:,}")
stats[1].metric("HTML features", len(FEATURE_NAMES))
stats[2].metric("Models compared", len(MODEL_DICT))

# Scan input
with st.form("scan"):
    url = st.text_input("Website URL", placeholder="https://example.com")
    submitted = st.form_submit_button("Scan", type="primary", width="stretch")

if submitted:
    if not url:
        st.error("Please enter a URL.")
    elif choice == URLSCAN_OPTION:
        st.session_state.pop("result", None)
        if not api_key:
            st.error("Please enter your Urlscan.io API key in the sidebar.")
        else:
            check_urlscan_api(api_key, normalize_url(url))
    else:
        with st.spinner("Fetching page and extracting features..."):
            st.session_state["result"] = scan_with_model(url, choice)

result = st.session_state.get("result")

tab_result, tab_features, tab_about = st.tabs(["Scan result", "Extracted features", "About the project"])

with tab_result:
    if result is None:
        st.info("Enter a URL and press Scan to check it.")
    elif "error" in result:
        st.error(result["error"])
    else:
        render_verdict(result)

with tab_features:
    if result is None or "error" in result:
        st.info("Run an ML scan to see the features extracted from the page.")
    else:
        features_df = pd.DataFrame({"Feature": FEATURE_NAMES, "Value": result["vector"]})
        st.dataframe(features_df, hide_index=True, width="stretch", height=420)

with tab_about:
    st.subheader("Approach")
    st.write("The app fetches a page, parses its HTML with BeautifulSoup, and turns it into a vector of "
             f"{len(FEATURE_NAMES)} structural features (forms, password fields, hidden elements, links, scripts, "
             "and similar). Supervised models trained on labelled pages then classify it as legitimate or phishing. "
             "Urlscan.io can be used as a second, external opinion.")
    st.subheader("Dataset")
    st.write(f"{n_legit + n_phish:,} websites collected in October 2022: {n_legit:,} legitimate (Tranco list) "
             f"and {n_phish:,} phishing (PhishTank). Label 1 = phishing, 0 = legitimate.")
    st.dataframe(ml.legitimate_df.head(10), width="stretch")
    st.subheader("Publication")
    st.markdown(f"Published at the IEEE International Conference on Machine Learning and Autonomous Systems "
                f"(ICMLAS 2025). [Read the paper]({PAPER_URL})")

st.markdown(f'<div class="footer">Built by Jasper Surya Chowdary Kota · '
            f'<a href="{GITHUB_URL}">GitHub</a> · <a href="{PORTFOLIO_URL}">Portfolio</a></div>',
            unsafe_allow_html=True)

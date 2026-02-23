import streamlit as st
import pandas as pd
import numpy as np
import joblib
import keras
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Card Customer Clustering",
    page_icon="💳",
    layout="wide"
)

# --- 🛠️ FIX: Keras Version Mismatch Workaround 🛠️ ---
# Bu sınıf, "quantization_config" hatasını yutmak için oluşturuldu.
# Keras 3 modelini Keras 2/Legacy ortamında açmaya çalışırken oluşan hatayı engeller.
class SafeDense(Dense):
    def __init__(self, *args, **kwargs):
        # 'quantization_config' parametresi gelirse sil, gerisini aynen ilet
        if 'quantization_config' in kwargs:
            kwargs.pop('quantization_config')
        super().__init__(*args, **kwargs)

# Modeli yüklerken 'Dense' katmanını bizim 'SafeDense' ile değiştireceğiz.
custom_objects_dict = {'Dense': SafeDense}

# --- PERSONA DEFINITIONS (English) ---
CLUSTER_INFO = {
    0: {
        "title": "Balanced & Moderate User ⚖️",
        "description": "Uses both purchases and cash advances in a balanced way. Represents a stable and mid-level activity profile."
    },
    1: {
        "title": "Conservative & Passive User 🛡️",
        "description": "Lowest usage frequency, low limits, and minimal spending. Likely keeps the card for security/emergency purposes."
    },
    2: {
        "title": "Cash Advance Oriented & Indebted 💸",
        "description": "Uses cash advance significantly more than purchases. Struggles with revolving debt and carries higher risk."
    },
    3: {
        "title": "High-Spending Elite Customer 💎",
        "description": "Highest spending, credit limit, and payment power. The bank's most valuable premium customer segment."
    }
}

# --- Load Models & Assets ---
@st.cache_resource
def load_assets():
    # Kodun çalıştığı klasörü bul (src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Dosya yolları (Kod ile aynı klasörde arar)
    paths = {
        'encoder': os.path.join(current_dir, 'models', 'encoder_model.h5'),
        'scaler': os.path.join(current_dir, 'models', 'scaler.joblib'),
        'imputer': os.path.join(current_dir, 'models', 'imputer.joblib'),
        'kmeans': os.path.join(current_dir, 'models', 'kmeans.joblib'),
        'pca': os.path.join(current_dir, 'models', 'pca.joblib')
    }

    try:
        # HATA DÜZELTME: custom_objects kullanarak parametre hatasını engelliyoruz
        encoder = keras.models.load_model(
            paths['encoder'], 
            custom_objects=custom_objects_dict, 
            compile=False
        )
        
        scaler = joblib.load(paths['scaler'])
        imputer = joblib.load(paths['imputer'])
        kmeans = joblib.load(paths['kmeans'])
        pca = joblib.load(paths['pca'])
        return encoder, scaler, imputer, kmeans, pca
    except Exception as e:
        st.error("🚨 SYSTEM ERROR: Models could not be loaded!")
        st.error(f"Details: {e}")
        st.warning("Please verify your TensorFlow/Keras versions or try re-saving the model.")
        return None, None, None, None, None

encoder, scaler, imputer, kmeans, pca_model = load_assets()

# --- App Interface ---
st.title("🤖 Credit Card Customer Clustering & Persona Analysis")
st.markdown("This application segments customers based on their transaction history.")

# --- Persona Info Cards ---
st.info("ℹ️ **Identified Customer Profiles (Personas):**")
cols = st.columns(4)
for i, (cluster_id, info) in enumerate(CLUSTER_INFO.items()):
    with cols[i]:
        st.markdown(f"##### {info['title']}")
        st.caption(info['description'])
st.divider()

# --- Sidebar ---
st.sidebar.header("1. Data Upload")
uploaded_file = st.sidebar.file_uploader("Choose a CSV File", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("2. Use Sample Data")

# --- 🛠️ FIX: Robust Path Finder for Data 🛠️ ---
def find_data_path():
    filename = 'CCGENERAL.csv'
    current_dir = os.path.dirname(os.path.abspath(__file__)) # .../src
    project_root = os.path.dirname(current_dir)            # .../Prj_Clustering2
    
    # Aranacak tüm olası yollar (Sırayla kontrol edilir)
    possible_paths = [
        os.path.join(current_dir, 'data', filename),      # src/data/CC...
        os.path.join(current_dir, filename),              # src/CC...
        os.path.join(project_root, 'data', filename),     # Prj/data/CC...
        os.path.join(project_root, filename),             # Prj/CC...
        # Hugging Face özel yolu (bazen gerekebilir)
        os.path.join('/app/data', filename) 
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

if st.sidebar.button("Load Sample Dataset"):
    found_path = find_data_path()
    
    if found_path:
        try:
            sample_df = pd.read_csv(found_path).sample(100, random_state=42)
            uploaded_file = "sample" 
            st.session_state['df'] = sample_df
            st.sidebar.success(f"Loaded from: ...{found_path[-20:]}") # Yolun son kısmını göster
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    else:
        st.sidebar.error("❌ Sample data file (CC GENERAL...) not found in any expected folder.")
        st.sidebar.info("Check if the file is in 'src/data' or 'data' folder.")


# --- Main Logic ---
if uploaded_file is not None:
    if uploaded_file == "sample":
        if 'df' in st.session_state:
            df_input = st.session_state['df'].copy()
        else:
            st.warning("Please reload data.")
            st.stop()
    else:
        df_input = pd.read_csv(uploaded_file)

    st.subheader("1. Raw Data Preview")
    st.dataframe(df_input.head())

    # --- Prediction Function ---
    def predict_clusters(df):
        if encoder is None: return None
        
        original_data = df.copy()
        if 'CUST_ID' in df.columns:
            df_proc = df.drop('CUST_ID', axis=1)
        else:
            df_proc = df.copy()
            
        try:
            # scikit-learn version mismatch compatibility fix
            if not hasattr(imputer, '_fill_dtype'):
                imputer._fill_dtype = np.float64
                
            df_imputed = pd.DataFrame(imputer.transform(df_proc), columns=df_proc.columns)
            df_scaled = scaler.transform(df_imputed)
            encoded_features = encoder.predict(df_scaled)
            clusters = kmeans.predict(encoded_features)
            pca_features = pca_model.transform(encoded_features)

            df_results = original_data.copy()
            df_results['CLUSTER'] = clusters
            df_results['PERSONA'] = df_results['CLUSTER'].map(lambda x: CLUSTER_INFO.get(x, {}).get('title', f'Cluster {x}'))
            df_results['PC1'] = pca_features[:, 0]
            df_results['PC2'] = pca_features[:, 1]
            return df_results
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            return None

    # --- Execute ---
    if encoder:
        with st.spinner('Analyzing...'):
            results_df = predict_clusters(df_input)

        if results_df is not None:
            st.success("✅ Analysis Completed!")

            # --- Visual Analysis ---
            st.header("2. Visual Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Persona Distribution")
                persona_counts = results_df['PERSONA'].value_counts()
                
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                sns.barplot(x=persona_counts.values, y=persona_counts.index, ax=ax1, palette='viridis')
                ax1.set_xlabel("Count")
                st.pyplot(fig1)

            with col2:
                st.subheader("Segment Map (PCA)")
                
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x='PC1', y='PC2', hue='PERSONA', data=results_df, palette='viridis', s=100, alpha=0.8, ax=ax2)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                st.pyplot(fig2)

            # --- Table ---
            st.header("3. Detailed List")
            cols = list(results_df.columns)
            priority = ['CUST_ID', 'PERSONA', 'CLUSTER']
            final_cols = [c for c in priority if c in cols] + [c for c in cols if c not in priority and c not in ['PC1', 'PC2']]
            st.dataframe(results_df[final_cols])
            
            # Download
            csv = results_df[final_cols].to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "results.csv", "text/csv")
    else:
        st.error("System halted: Models not loaded.")
else:
    st.info("👈 Waiting for data upload...")
# app.py (Corrected)

import streamlit as st
import numpy as np
import pandas as pd
import ast
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import cv2  # Standard OpenCV import
from ingredients import KEY_INGREDIENTS  # Import the dictionary from the new file

# --- File Paths (Constants) ---
BERT_MODEL_PATH = 'models/bert_model.h5'
IMAGE_MODEL_PATH = 'models/cnn_model.h5'
TOKENIZER_PATH = 'assets/tokenizer.pkl'
LABELS_PATH = 'assets/labels.pkl'
DATASET_PATH = "final_product_database.csv"

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="DermaClust AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Models for Performance ---
@st.cache_resource
def load_models_and_assets():
    """Load all necessary models and assets."""
    bert_model = load_model(BERT_MODEL_PATH)
    image_model = load_model(IMAGE_MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(LABELS_PATH, 'rb') as f:
        labels = pickle.load(f)
    return bert_model, image_model, tokenizer, labels

@st.cache_data
def load_product_data():
    """Load and preprocess the product dataset."""
    df = pd.read_csv(DATASET_PATH)
    df['ingredients'] = df['clean_ingreds'].apply(lambda x: ast.literal_eval(x.lower()))
    df['ingredients_text'] = df['ingredients'].apply(lambda x: ' '.join(x))

    def assign_labels_by_ingredient(ingredient_list):
        assigned = set()
        for label, keywords in KEY_INGREDIENTS.items():
            for keyword in keywords:
                if keyword in ingredient_list:
                    assigned.add(label)
        if not assigned: return ['normal']
        return list(assigned)

    df['label_tags'] = df['ingredients'].apply(assign_labels_by_ingredient)
    return df  # <-- ERROR FIXED: Added return statement

# --- Recommendation Logic ---
def recommend_filtered_products(predicted_skin_type, all_products_df, top_k=5):
    filtered_df = all_products_df[all_products_df['label_tags'].apply(lambda tags: predicted_skin_type in tags)].copy()
    if filtered_df.empty: return []
    
    tokenizer = load_models_and_assets()[2] # Safely get tokenizer
    bert_model = load_models_and_assets()[0] # Safely get bert_model
    labels = load_models_and_assets()[3] # Safely get labels
    
    filtered_sequences = tokenizer.texts_to_sequences(filtered_df['ingredients_text'])
    filtered_X = pad_sequences(filtered_sequences, maxlen=100)
    product_vectors = bert_model.predict(filtered_X)
    
    predicted_skin_type_index = labels.index(predicted_skin_type)
    product_scores = product_vectors[:, predicted_skin_type_index]
    sorted_idx = np.argsort(product_scores)[::-1]
    
    recommendations = []
    for i in sorted_idx[:top_k]:
        recommendations.append({
            "name": filtered_df.iloc[i]['product_name'],
            "score": product_scores[i]
        })
    return recommendations

# --- Load all data ---
bert_model, image_model, tokenizer, labels = load_models_and_assets()
df = load_product_data()

# --- SIDEBAR ---
st.sidebar.title("About DermaClust AI")
st.sidebar.info(
    "This is an AI-powered skincare recommendation system. "
    "It uses a Convolutional Neural Network (CNN) to analyze your skin type from an image "
    "and a Transformer-based model to understand product ingredients, providing you with personalized recommendations."
)
num_classes = image_model.output_shape[-1]
st.sidebar.write(f"✅ Models loaded successfully.")
st.sidebar.write(f"🧠 Image model is trained for {num_classes} skin types.")

# --- MAIN PAGE LAYOUT ---
st.title("🌿 DermaClust: Your Personal AI Skincare Advisor")
st.write("Choose your preferred method to provide a skin image. Our AI will then analyze it and recommend the best products for you.")

tab1, tab2 = st.tabs(["📁 Upload an Image", "📸 Use Camera"])

# --- Tab 1: File Uploader ---
with tab1:
    st.header("Upload a Picture of Your Skin")
    uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

# --- Tab 2: Camera Input ---
with tab2:
    st.header("Take a Picture with Your Camera")
    camera_file = st.camera_input("Point the camera at your face and click the button", label_visibility="collapsed")

# --- Unified Logic to Process the Image ---
image_to_process = camera_file if camera_file is not None else uploaded_file
results_placeholder = st.empty()
# In app.py

if image_to_process is not None:
    with results_placeholder.container():
        st.header("🔬 Analysis & Recommendations")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image_to_process, caption="Image to be Analyzed", use_column_width=True)

        with col2:
           # Corrected code block for app.py

            try:
                # Face Detection with the correct 'cv2' prefix
                file_bytes = np.asarray(bytearray(image_to_process.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1) # Changed cv to cv2
                gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY) # Changed cv to cv2
                
                face_cascade = cv2.CascadeClassifier(r"C:\Users\yadav\OneDrive\Documents\Dermaclust Project\classifiers\haarcascade_frontalface_default.xml") # Changed cv to cv2
                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) == 0:
                    st.error("No face detected. Please provide a clear, forward-facing picture.")
                else:
                    st.success(f"✅ Face detected! Analyzing skin type...")
                    image_to_process.seek(0)
                    
                    # Preprocess and predict
                    img = image.load_img(image_to_process, target_size=(128, 128))
                    img_array = image.img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    with st.spinner('Analyzing skin type...'):
                        prediction = image_model.predict(img_array)[0]
                        # --- NEW: GET TOP TWO PREDICTIONS ---
                        # Get the indices of the top two predictions by sorting them
                        top_two_indices = np.argsort(prediction)[::-1][:2]
                        primary_skin_type = labels[top_two_indices[0]]
                        secondary_skin_type = labels[top_two_indices[1]]
                        
                    st.success(f"**Primary Skin Type: {primary_skin_type.capitalize()}**")
                    st.info(f"**Secondary Concern: {secondary_skin_type.capitalize()}**")
                    
                    # --- NEW: GET RECOMMENDATIONS FOR BOTH ---
                    with st.spinner('Finding the best products...'):
                        primary_recs = recommend_filtered_products(primary_skin_type, df)
                        secondary_recs = recommend_filtered_products(secondary_skin_type, df)
                    
                    # Display Primary Recommendations
                    st.subheader(f"Top Recommendations for {primary_skin_type.capitalize()} Skin")
                    if not primary_recs:
                        st.warning("Sorry, no specific products found.")
                    else:
                        for rec in primary_recs:
                            st.write(f"**{rec['name']}** (Match Score: {rec['score']:.2f})")
                            
                    # Display Secondary Recommendations
                    st.subheader(f"Also Recommended for {secondary_skin_type.capitalize()} Skin")
                    if not secondary_recs:
                        st.warning("Sorry, no specific products found.")
                    else:
                        for rec in secondary_recs:
                            st.write(f"**{rec['name']}** (Match Score: {rec['score']:.2f})")

            except Exception as e:
                st.error(f"An error occurred: {e}. The file might be corrupted.")
else:
    results_placeholder.info("Please upload an image or use the camera to get started.")
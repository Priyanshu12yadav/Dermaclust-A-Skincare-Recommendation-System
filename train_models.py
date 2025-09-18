# train_models.py (Updated)

import pandas as pd
import ast
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, MultiHeadAttention, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout

# --- IMPORT a single source of truth for ingredients ---
from ingredients import KEY_INGREDIENTS

print("--- Starting Model Training Process ---")

# --- STEP 1: LOAD AND PREPARE TEXT DATA ---
print("Step 1/5: Loading and preparing text data...")
try:
    df = pd.read_csv("final_product_database.csv")
    df['ingredients'] = df['clean_ingreds'].apply(lambda x: ast.literal_eval(x.lower()))
    df['ingredients_text'] = df['ingredients'].apply(lambda x: ' '.join(x))

    # --- INGREDIENT-BASED LABELING LOGIC ---
    def assign_labels_by_ingredient(ingredient_list):
        assigned = set()
        for label, keywords in KEY_INGREDIENTS.items():
            for keyword in keywords:
                if keyword in ingredient_list:
                    assigned.add(label)
        if not assigned:
            return ['normal']
        return list(assigned)

    df['label_tags'] = df['ingredients'].apply(assign_labels_by_ingredient)
    # Define the master list of all possible labels
    labels = ['normal', 'oily', 'dry', 'acne', 'wrinkles']
    
    print("Text data loaded and labeled using the complete ingredient list.")
except FileNotFoundError:
    print("\nERROR: 'final_product_database.csv' not found.")
    print("Please make sure the Excel file is in the same folder as this script.")
    exit()

# --- STEP 2: TOKENIZE TEXT AND TRAIN BERT-STYLE MODEL ---
print("\nStep 2/5: Building and training the ingredients (BERT-style) model...")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['ingredients_text'])
sequences = tokenizer.texts_to_sequences(df['ingredients_text'])
X = pad_sequences(sequences, maxlen=100)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['label_tags'])

input_layer = Input(shape=(100,))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64)(input_layer)
attention_output = MultiHeadAttention(num_heads=2, key_dim=64)(embedding_layer, embedding_layer)
add_norm1 = LayerNormalization()(attention_output + embedding_layer)
ff = Dense(64, activation='relu')(add_norm1)
add_norm2 = LayerNormalization()(ff + add_norm1)
flat_layer = Flatten()(add_norm2)
output = Dense(len(labels), activation='sigmoid')(flat_layer)
bert_model = Model(inputs=input_layer, outputs=output)
bert_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
bert_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32)
print("Ingredients model trained successfully.")

# --- STEP 3: TRAIN SKIN ANALYZER CNN MODEL ---
print("\nStep 3/5: Building and training the skin image (CNN) model...")

img_size = (128, 128)
image_path = os.path.join("dataset", "faceskin")

if not os.path.exists(image_path):
    print(f"\nERROR: Image directory not found at '{image_path}'")
    print("Please ensure your 'faceskin' folder is inside a 'dataset' folder.")
    exit()



# NEW, IMPROVED LINE:
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,      # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally
    height_shift_range=0.2, # randomly shift images vertically
    shear_range=0.2,        # shear transformation
    zoom_range=0.2,         # randomly zoom in
    horizontal_flip=True,   # randomly flip images
    fill_mode='nearest'
)
train_gen = datagen.flow_from_directory(
    image_path, # Use the variable you already defined
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val_gen = datagen.flow_from_directory(
    image_path, # Also use the variable here
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

num_classes = train_gen.num_classes
image_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
image_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
image_model.fit(train_gen, validation_data=val_gen, epochs=15)
print("Skin image model trained successfully.")

# --- STEP 4: SAVE EVERYTHING ---
print("\nStep 4/5: Saving all models and assets...")

os.makedirs('models', exist_ok=True)
os.makedirs('assets', exist_ok=True)

bert_model.save('models/bert_model.h5')
image_model.save('models/cnn_model.h5')

with open('assets/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('assets/labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

print("All files saved successfully.")

# --- STEP 5: FINISH ---
print("\n--- Model Training Process Finished ---")
print("You can now run your Streamlit app with the command: streamlit run app.py")
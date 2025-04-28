import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import json
import pickle

TILES_DIR = "King Domino dataset/King Domino dataset/Extracted_Tiles"
LABELS_FILE = "Excel+JSON/tile_labels_mapping.json"
MODEL_FILE = "kingdomino_terrain_model.pkl"

def load_data():
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"Labels-filen {LABELS_FILE} blev ikke fundet. Kør labels.py først.")
    
    with open(LABELS_FILE, 'r') as f:
        labels_data = json.load(f)
    
    images = []
    terrain_labels = []
    unique_terrains = set()
    
    for board_name, tiles in labels_data.items():
        for tile_pos, tile_info in tiles.items():
            file_path = os.path.join(TILES_DIR, tile_info["filename"])
            
            if not os.path.exists(file_path):
                print(f"Advarsel: Filen {file_path} blev ikke fundet. Springer over.")
                continue
            
            img = cv2.imread(file_path)
            if img is None:
                print(f"Advarsel: Kunne ikke indlæse billedet {file_path}. Springer over.")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            terrain = tile_info["terrain"]
            if terrain in ["Unknown", "Home", "Table"]:
                continue
            
            images.append(img)
            terrain_labels.append(terrain)
            unique_terrains.add(terrain)
    
    terrain_classes = {terrain: i for i, terrain in enumerate(sorted(unique_terrains))}
    labels = [terrain_classes[terrain] for terrain in terrain_labels]
    
    return images, np.array(labels), terrain_classes

def extract_hsv_histogram(image, bins=32):
    try:
        hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    except:
        print("Advarsel: Kunne ikke konvertere til HSV")
        return np.zeros(bins * 3)
    
    hist_h = cv2.calcHist([hsv_img], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [bins], [0, 256])
    
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    return np.concatenate([hist_h, hist_s, hist_v])

def extract_texture_histogram(image, bins=9):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except:
        print("Advarsel: Kunne ikke konvertere til gråtone")
        return np.zeros(bins)
    
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi) % 180
    
    hist = np.zeros(bins)
    bin_width = 180 / bins
    
    for i in range(bins):
        bin_start = i * bin_width
        bin_end = (i + 1) * bin_width
        
        mask = ((direction >= bin_start) & (direction < bin_end))
        hist[i] = np.sum(magnitude[mask])
    
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist

def extract_features(image):
    hsv_hist = extract_hsv_histogram(image)
    texture_hist = extract_texture_histogram(image)
    combined_features = np.concatenate([hsv_hist, texture_hist])
    return combined_features

def extract_features_batch(images):
    features = []
    for i, image in enumerate(images):
        if i % 100 == 0:
            print(f"Udtrækker features for billede {i+1}/{len(images)}...")
        features.append(extract_features(image))
    return np.array(features)

class TerrainClassifier:
    def __init__(self, lda=None, knn=None, terrain_classes=None):
        self.lda = lda
        self.knn = knn
        self.terrain_classes = terrain_classes
        self.is_fitted = (lda is not None and knn is not None)
    
    def fit(self, X, y):
        n_components = min(len(np.unique(y)), X.shape[1]) - 1
        
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = self.lda.fit_transform(X, y)
        
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.knn.fit(X_lda, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Modellen er ikke trænet endnu.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_lda = self.lda.transform(X)
        return self.knn.predict(X_lda)
    
    def predict_terrain(self, X):
        if self.terrain_classes is None:
            raise ValueError("Terrain classes mapping er ikke tilgængelig.")
        
        y_pred = self.predict(X)
        terrain_names = {v: k for k, v in self.terrain_classes.items()}
        return [terrain_names[label] for label in y_pred]

def save_model(model, file_path=MODEL_FILE):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model gemt til: {file_path}")

def load_model(file_path=MODEL_FILE):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Modelfilen {file_path} findes ikke.")
    
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model indlæst fra: {file_path}")
    return model

def main():
    print("Indlæser data...")
    images, labels, terrain_classes = load_data()
    
    print(f"Indlæst {len(images)} billeder med {len(set(labels))} forskellige terrænklasser.")
    print("Terrænklasser:", {k: v for k, v in sorted(terrain_classes.items(), key=lambda x: x[1])})
    
    print("Udtrækker HSV og tekstur features...")
    features = extract_features_batch(images)
    print(f"Udtrukket features med form: {features.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print("Træner TerrainClassifier model (LDA + KNN)...")
    model = TerrainClassifier(terrain_classes=terrain_classes)
    model.fit(X_train, y_train)
    
    print("Evaluerer model...")
    y_pred = model.predict(X_test)
    
    print("Klassifikationsrapport:")
    print(classification_report(y_test, y_pred))
    
    print("Konfusionsmatrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    terrain_names = {v: k for k, v in terrain_classes.items()}
    terrain_labels = [terrain_names[i] for i in range(len(terrain_names))]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Konfusionsmatrix')
    plt.colorbar()
    tick_marks = np.arange(len(terrain_labels))
    plt.xticks(tick_marks, terrain_labels, rotation=90)
    plt.yticks(tick_marks, terrain_labels)
    plt.tight_layout()
    plt.ylabel('Faktisk label')
    plt.xlabel('Forudsagt label')
    plt.savefig('confusion_matrix.png')
    
    print("Resultater gemt til confusion_matrix.png")
    
    print("Gemmer model...")
    save_model(model)
    
    print("Træning og evaluering afsluttet. Model er klar til brug")

if __name__ == "__main__":
    main()
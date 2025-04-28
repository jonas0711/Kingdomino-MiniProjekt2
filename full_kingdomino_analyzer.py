import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import cv2
import pickle
import re
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Import required components from model.py
from model import (
    TerrainClassifier, 
    load_model as load_model_from_model_py, 
    extract_features, 
    extract_hsv_histogram, 
    extract_texture_histogram,
    load_data
)

# Constants
TERRAIN_MODEL_FILE = "kingdomino_terrain_model.pkl"
CROWN_TEMPLATES_DIR = "Crown_Templates"
TERRAIN_CATEGORIES_DIR = "KingDominoDataset/TerrainCategories"
TERRAIN_TYPES = ["Field", "Forest", "Grassland", "Lake", "Mine", "Swamp"]
CROWN_TEMPLATE_NAMES = ["Up", "Down", "Left", "Right"]
RESULTS_DIR = "Model_Evaluation_Results"

# Ensure output directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_terrain_model(model_path=TERRAIN_MODEL_FILE):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_full_path = model_path if os.path.isabs(model_path) else os.path.join(current_dir, model_path)
        
        print(f"Forsøger at indlæse model fra: {model_full_path}")
        with open(model_full_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Terrænklassifikationsmodel indlæst fra {model_full_path}")
        return model
    except FileNotFoundError:
        print(f"Fejl: Modelfilen blev ikke fundet på {model_full_path}.")
        return None
    except Exception as e:
        print(f"Fejl ved indlæsning af model: {e}")
        return None

def load_crown_templates(templates_dir=CROWN_TEMPLATES_DIR):
    template_paths = [
        os.path.join(templates_dir, "Crown_up.png"),
        os.path.join(templates_dir, "Crown_down.png"),
        os.path.join(templates_dir, "Crown_left.png"),
        os.path.join(templates_dir, "Crown_right.png")
    ]
    
    templates = []
    for path in template_paths:
        try:
            template = cv2.imread(path)
            if template is None:
                 raise FileNotFoundError(f"Kunne ikke indlæse template fra {path}")
                
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            templates.append(template_gray)
        except FileNotFoundError as e:
             print(f"Advarsel: {e}")
        except Exception as e:
            print(f"Advarsel ved indlæsning af template {path}: {e}")

    if len(templates) != len(template_paths):
         print("Advarsel: Ikke alle crown templates blev indlæst.")
         
    return templates, CROWN_TEMPLATE_NAMES[:len(templates)]

def extract_info_from_filename(filename):
    match = re.search(r'_([A-Za-z]+)_(\d+)crowns\.png$', filename)
    if match:
        terrain_type = match.group(1)
        crowns = int(match.group(2))
        return terrain_type, crowns
    return None, 0

def detect_crowns_in_tile(tile_image, templates, template_names, threshold=0.6):
    if len(tile_image.shape) == 3 and tile_image.shape[2] == 3:
         try:
             tile_gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
         except:
              try:
                  tile_gray = cv2.cvtColor(tile_image, cv2.COLOR_RGB2GRAY)
              except:
                  if tile_image.ndim == 2:
                       tile_gray = tile_image
                  else:
                       print(f"Advarsel: Uventet billedformat for kronedetektering {tile_image.shape}")
                       tile_gray = tile_image[:,:,0]
    elif tile_image.ndim == 2:
         tile_gray = tile_image
    else:
         print(f"Advarsel: Uventet billedformat for kronedetektering {tile_image.shape}")
         tile_gray = tile_image[:,:,0]

    best_matches = []
    
    for template, template_name in zip(templates, template_names):
        if template is None:
            continue
        if tile_gray.dtype != np.uint8:
            tile_gray = cv2.convertScaleAbs(tile_gray)
        if template.dtype != np.uint8:
             template = cv2.convertScaleAbs(template)

        if tile_gray.shape[0] < template.shape[0] or tile_gray.shape[1] < template.shape[1]:
            continue

        result = cv2.matchTemplate(tile_gray, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        for pt in zip(*locations[::-1]):
            match_value = result[pt[1], pt[0]]
            best_matches.append({
                'value': match_value,
                'position': pt,
                'template': template_name,
                'template_size': template.shape
            })
    
    best_matches.sort(key=lambda x: x['value'], reverse=True)
    
    final_matches = []
    while best_matches:
        best_match = best_matches.pop(0)
        final_matches.append(best_match)
        
        non_overlapping_matches = []
        for match in best_matches:
            best_center_x = best_match['position'][0] + best_match['template_size'][1] // 2
            best_center_y = best_match['position'][1] + best_match['template_size'][0] // 2
            
            match_center_x = match['position'][0] + match['template_size'][1] // 2
            match_center_y = match['position'][1] + match['template_size'][0] // 2
            
            distance = np.sqrt((best_center_x - match_center_x)**2 + (best_center_y - match_center_y)**2)
            
            if distance > min(best_match['template_size'][0], best_match['template_size'][1]) // 2:
                non_overlapping_matches.append(match)
        
        best_matches = non_overlapping_matches

    crown_count = len(final_matches)
    centroids = [(match['position'][0] + match['template_size'][1] // 2,
                  match['position'][1] + match['template_size'][0] // 2)
                 for match in final_matches]

    return crown_count, centroids

def classify_tile(tile_image, terrain_model, templates, template_names, crown_threshold=0.6, visualize=False):
    terrain_type = "Unknown"
    try:
        if isinstance(tile_image, np.ndarray):
            if len(tile_image.shape) == 3 and tile_image.shape[2] == 3:
                 try:
                    rgb_image = cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB)
                 except:
                    rgb_image = tile_image
            elif len(tile_image.shape) == 3 and tile_image.shape[2] == 4:
                 rgb_image = cv2.cvtColor(tile_image, cv2.COLOR_RGBA2RGB)
            elif tile_image.ndim == 2:
                 rgb_image = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2RGB)
            else:
                 rgb_image = tile_image[:,:,:3]
        else:
            try:
                rgb_image = np.array(tile_image)
                if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
                     try:
                        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                     except:
                        pass
                elif len(rgb_image.shape) == 3 and rgb_image.shape[2] == 4:
                     rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)
                elif rgb_image.ndim == 2:
                     rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
                else:
                     rgb_image = rgb_image[:,:,:3]
            except Exception as conv_e:
                return "Unknown", 0, []

        features = extract_features(rgb_image)
        features_2d = features.reshape(1, -1)
        terrain_type = terrain_model.predict_terrain(features_2d)[0]
    except Exception as e:
        terrain_type = "Unknown"

    crown_count = 0
    centroids = []
    try:
        crown_count, centroids = detect_crowns_in_tile(
            tile_image, templates, template_names, crown_threshold
        )
    except Exception as e:
        crown_count = 0
        centroids = []

    if visualize and tile_image is not None:
        plt.figure(figsize=(10, 5))
        
        if len(tile_image.shape) == 3 and tile_image.shape[2] == 3:
             try:
                display_image = cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB)
             except:
                display_image = tile_image
        elif len(tile_image.shape) == 3 and tile_image.shape[2] == 4:
             display_image = cv2.cvtColor(tile_image, cv2.COLOR_RGBA2RGB)
        elif tile_image.ndim == 2:
             display_image = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2RGB)
        else:
             display_image = tile_image

        plt.subplot(1, 2, 1)
        plt.imshow(display_image)
        plt.title("Originalt tile")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        result_img = np.copy(display_image)
        
        for (cx, cy) in centroids:
             cv2.circle(result_img, (cx, cy), 5, (0, 255, 0), -1)

        plt.imshow(result_img)
        plt.title(f"Klassificeret: {terrain_type}, Kroner: {crown_count}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return terrain_type, crown_count, centroids

def evaluate_on_test_data(terrain_model, templates, template_names, visualize_samples=5):
    """Evaluerer modellen på test data (20% split fra original træning)"""
    print("\nIndlæser original data til at recreate train/test split...")
    images, labels, terrain_classes = load_data()
    
    print(f"Indlæst {len(images)} billeder med {len(set(labels))} forskellige terrænklasser.")
    
    # Udtræk features for alle billeder
    print("Udtrækker features...")
    features = []
    for i, image in enumerate(images):
        if i % 100 == 0:
            print(f"  Bearbejder billede {i+1}/{len(images)}...")
        features.append(extract_features(image))
    features = np.array(features)
    
    # Recreate samme train/test split som i træningen
    print("Recreating train/test split (80/20)...")
    _, X_test, _, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Evaluerer på {len(X_test)} test-samples...")
    
    # Konverter y_test til terrænnavne
    terrain_names = {v: k for k, v in terrain_classes.items()}
    y_test_terrain = [terrain_names[label] for label in y_test]
    
    # Predict med model
    y_pred_terrain = terrain_model.predict_terrain(X_test)
    
    # Crown detection resultater
    y_true_crowns = []
    y_pred_crowns = []
    
    # Evaluer crown detection på test data
    print("Evaluerer kronedetektering...")
    
    terrain_samples = {}
    for i, (image, true_terrain) in enumerate(zip(images, [terrain_names[label] for label in labels])):
        if true_terrain not in terrain_samples:
            terrain_samples[true_terrain] = []
        terrain_samples[true_terrain].append(image)
    
    # Test på et mindre subset for kronedetektering (da vi ikke har ground truth for alle kronetællinger)
    test_terrain_images = {}
    test_crown_count = 0
    
    # Evaluer kun på et subsæt af billeder i TerrainCategories
    for terrain_type in TERRAIN_TYPES:
        terrain_dir = os.path.join(TERRAIN_CATEGORIES_DIR, terrain_type)
        if not os.path.exists(terrain_dir):
            print(f"Advarsel: Mappen {terrain_dir} findes ikke.")
            continue
        
        files = [f for f in os.listdir(terrain_dir) if f.endswith('.png')]
        test_terrain_images[terrain_type] = []
        
        for i, file in enumerate(files):
            file_path = os.path.join(terrain_dir, file)
            
            if not os.path.exists(file_path):
                continue
                
            tile = cv2.imread(file_path)
            if tile is None:
                continue
            
            actual_terrain, actual_crown_count = extract_info_from_filename(file)
            
            if actual_terrain is None:
                actual_terrain = terrain_type
            
            # Forventet terrain og crown count
            y_true_crowns.append(actual_crown_count)
            
            # Beregn terrain og crown med vores model
            predicted_terrain, predicted_crown_count, _ = classify_tile(
                tile, terrain_model, templates, template_names,
                crown_threshold=0.6, visualize=(i < visualize_samples)
            )
            
            y_pred_crowns.append(predicted_crown_count)
            test_crown_count += 1
    
    # Beregn metrics
    terrain_accuracy = accuracy_score(y_test_terrain, y_pred_terrain)
    
    # For kroner bruger vi exact match accuracy på det subset vi evaluerer
    crown_correct = sum(1 for true, pred in zip(y_true_crowns, y_pred_crowns) if true == pred)
    crown_accuracy = crown_correct / len(y_true_crowns) if y_true_crowns else 0
    
    # Beregn også metrics pr terræntype
    terrain_metrics = {}
    for terrain in sorted(set(y_test_terrain)):
        indices = [i for i, t in enumerate(y_test_terrain) if t == terrain]
        terrain_true = [y_test_terrain[i] for i in indices]
        terrain_pred = [y_pred_terrain[i] for i in indices]
        terrain_accuracy = accuracy_score(terrain_true, terrain_pred)
        
        terrain_metrics[terrain] = {
            'count': len(terrain_true),
            'accuracy': terrain_accuracy
        }
    
    # Beregn crown detection metrics pr terræntype (for det subsæt vi har)
    crown_metrics_by_terrain = {}
    for terrain_type in TERRAIN_TYPES:
        terrain_dir = os.path.join(TERRAIN_CATEGORIES_DIR, terrain_type)
        if not os.path.exists(terrain_dir):
            continue
            
        files = [f for f in os.listdir(terrain_dir) if f.endswith('.png')]
        true_crowns = []
        pred_crowns = []
        
        for file in files:
            file_path = os.path.join(terrain_dir, file)
            tile = cv2.imread(file_path)
            if tile is None:
                continue
                
            actual_terrain, actual_crown_count = extract_info_from_filename(file)
            if actual_terrain is None:
                actual_terrain = terrain_type
                
            _, predicted_crown_count, _ = classify_tile(
                tile, terrain_model, templates, template_names, crown_threshold=0.6
            )
            
            true_crowns.append(actual_crown_count)
            pred_crowns.append(predicted_crown_count)
        
        if true_crowns:
            crown_correct = sum(1 for true, pred in zip(true_crowns, pred_crowns) if true == pred)
            crown_accuracy = crown_correct / len(true_crowns)
            
            crown_metrics_by_terrain[terrain_type] = {
                'crown_accuracy': crown_accuracy,
                'sample_count': len(true_crowns),
                'true_crown_count': sum(true_crowns),
                'pred_crown_count': sum(pred_crowns)
            }
    
    # Opret detaljeret rapport
    results = {
        'terrain_accuracy': terrain_accuracy,
        'crown_accuracy': crown_accuracy,
        'test_samples': len(y_test),
        'crown_test_samples': test_crown_count,
        'terrain_metrics': terrain_metrics,
        'crown_metrics_by_terrain': crown_metrics_by_terrain,
        'classification_report': classification_report(y_test_terrain, y_pred_terrain, output_dict=True)
    }
    
    # Plot confusion matrix for terrain type
    plot_confusion_matrix(
        y_test_terrain, 
        y_pred_terrain, 
        sorted(set(y_test_terrain)), 
        'Terrain Classification Confusion Matrix'
    )
    
    # Print resultater
    print("\n===== MODEL EVALUATION RESULTS =====")
    print(f"Terrain Classification Accuracy: {terrain_accuracy:.4f} ({len(y_test)} samples)")
    print(f"Crown Detection Accuracy: {crown_accuracy:.4f} ({test_crown_count} samples)")
    
    print("\nAccuracy pr. terræntype:")
    for terrain, metrics in sorted(terrain_metrics.items()):
        print(f"  {terrain}: {metrics['accuracy']:.4f} ({metrics['count']} samples)")
    
    print("\nKrone-accuracy pr. terræntype:")
    for terrain, metrics in sorted(crown_metrics_by_terrain.items()):
        print(f"  {terrain}: {metrics['crown_accuracy']:.4f} (Sand sum: {metrics['true_crown_count']}, Forudsagt sum: {metrics['pred_crown_count']})")
    
    # Gem resultater til JSON
    with open(os.path.join(RESULTS_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_confusion_matrix(actual, predicted, labels, title):
    cm = confusion_matrix(actual, predicted, labels=labels)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    
    # Normaliseret CM for bedre visualisering
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Tilføj labels til celler
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=8)
    
    plt.tight_layout()
    plt.ylabel('Sand Label')
    plt.xlabel('Forudsagt Label')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plt.close()

def main():
    # Indlæs model og templates
    print("Indlæser terrænklassifikationsmodel...")
    terrain_model = load_terrain_model()
    if terrain_model is None:
        print("Kunne ikke indlæse terrænklassifikationsmodel. Afslutter.")
        return
    
    print("Indlæser crown templates...")
    templates, template_names = load_crown_templates()
    if not templates:
        print("Kunne ikke indlæse crown templates. Afslutter.")
        return
    
    print("Evaluerer model på 20% test data fra det oprindelige train/test split...")
    evaluate_on_test_data(
        terrain_model, templates, template_names, 
        visualize_samples=3
    )
    
    print(f"\nEvalueringen er fuldført! Resultater er gemt i {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
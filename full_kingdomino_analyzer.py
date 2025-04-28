import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import cv2
import pickle
import re
import random
from sklearn.metrics import classification_report, confusion_matrix

# Import required components from model.py for deserialization and feature extraction
from model import TerrainClassifier, load_model as load_model_from_model_py, extract_features, extract_hsv_histogram, extract_texture_histogram

# Constants
TERRAIN_MODEL_FILE = "kingdomino_terrain_model.pkl"
CROWN_TEMPLATES_DIR = "KingDominoDataset/Crown_Templates"
TERRAIN_CATEGORIES_DIR = "KingDominoDataset/TerrainCategories"
TERRAIN_TYPES = ["Field", "Forest", "Grassland", "Lake", "Mine", "Swamp"]
CROWN_TEMPLATE_NAMES = ["Up", "Down", "Left", "Right"]

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
                 print(f"Advarsel: Uventet billedformat for terrænklassifikation {tile_image.shape}. Bruger første 3 kanaler.")
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
                     print(f"Advarsel: Uventet billedformat for terrænklassifikation efter konvertering til numpy {rgb_image.shape}. Bruger første 3 kanaler.")
                     rgb_image = rgb_image[:,:,:3]
            except Exception as conv_e:
                print(f"Fejl ved konvertering af billede til numpy/RGB: {conv_e}")
                return "Unknown", 0, []

        features = extract_features(rgb_image)
        features_2d = features.reshape(1, -1)
        terrain_type = terrain_model.predict_terrain(features_2d)[0]
    except Exception as e:
        print(f"Fejl under terrænklassifikation: {e}")
        terrain_type = "Unknown"

    crown_count = 0
    centroids = []
    try:
        crown_count, centroids = detect_crowns_in_tile(
            tile_image, templates, template_names, crown_threshold
        )
    except Exception as e:
        print(f"Fejl under kronedetektering: {e}")
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

def evaluate_on_sample(terrain_model, templates, template_names, samples_per_terrain=10, visualize=False):
    results = []
    actual_terrains = []
    predicted_terrains = []
    actual_crowns = []
    predicted_crowns = []
    
    for terrain_type in TERRAIN_TYPES:
        terrain_dir = os.path.join(TERRAIN_CATEGORIES_DIR, terrain_type)
        
        if not os.path.exists(terrain_dir):
            print(f"Advarsel: Mappen {terrain_dir} findes ikke.")
            continue
        
        files = [f for f in os.listdir(terrain_dir) if f.endswith('.png')]
        
        if len(files) > samples_per_terrain:
            files = random.sample(files, samples_per_terrain)
        
        print(f"\nEvaluerer på {len(files)} samples fra {terrain_type}...")
        
        for i, file in enumerate(files):
            file_path = os.path.join(terrain_dir, file)
            
            tile = cv2.imread(file_path)
            if tile is None:
                print(f"Kunne ikke indlæse {file_path}")
                continue
            
            actual_terrain, actual_crown_count = extract_info_from_filename(file)
            
            if actual_terrain is None:
                actual_terrain = terrain_type

            predicted_terrain, predicted_crown_count, _ = classify_tile(
                tile, terrain_model, templates, template_names,
                crown_threshold=0.6, visualize=(visualize and i == 0)
            )
            
            actual_terrains.append(actual_terrain)
            predicted_terrains.append(predicted_terrain)
            actual_crowns.append(actual_crown_count)
            predicted_crowns.append(predicted_crown_count)
            
            results.append({
                'filename': file,
                'actual_terrain': actual_terrain,
                'actual_crowns': actual_crown_count,
                'predicted_terrain': predicted_terrain,
                'predicted_crowns': predicted_crown_count,
                'terrain_correct': actual_terrain == predicted_terrain,
                'crowns_correct': actual_crown_count == predicted_crown_count,
                'fully_correct': (actual_terrain == predicted_terrain) and (actual_crown_count == predicted_crown_count)
            })

    total = len(results)
    terrain_correct = sum(1 for r in results if r['terrain_correct'])
    crowns_correct = sum(1 for r in results if r['crowns_correct'])
    fully_correct = sum(1 for r in results if r['fully_correct'])
    
    terrain_accuracy = terrain_correct / total if total > 0 else 0
    crowns_accuracy = crowns_correct / total if total > 0 else 0
    overall_accuracy = fully_correct / total if total > 0 else 0
    
    print("\n===== PERFORMANCE REPORT =====")
    print(f"Total samples: {total}")
    print(f"Terrain classification accuracy: {terrain_accuracy:.2f} ({terrain_correct}/{total})")
    print(f"Crown detection accuracy: {crowns_accuracy:.2f} ({crowns_correct}/{total})")
    print(f"Overall accuracy: {overall_accuracy:.2f} ({fully_correct}/{total})")
    
    print("\nTerrain Classification Report (sklearn):")
    try:
        all_possible_terrain_labels = sorted(list(set(actual_terrains + predicted_terrains)))
        print(classification_report(actual_terrains, predicted_terrains, labels=all_possible_terrain_labels, zero_division=0))
    except Exception as e:
        print(f"Fejl ved generering af klassifikationsrapport: {e}")
    
    print("\nCrown Detection Accuracy by Actual Crown Count:")
    crown_counts_stats = {}
    for r in results:
        count = r['actual_crowns']
        if count not in crown_counts_stats:
            crown_counts_stats[count] = {'total': 0, 'correct': 0}
        crown_counts_stats[count]['total'] += 1
        if r['crowns_correct']:
            crown_counts_stats[count]['correct'] += 1
    
    for count, stats in sorted(crown_counts_stats.items()):
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {count} kroner: {accuracy:.2f} ({stats['correct']}/{stats['total']})")
    
    return {'actual_terrains': actual_terrains, 'predicted_terrains': predicted_terrains,
            'actual_crowns': actual_crowns, 'predicted_crowns': predicted_crowns}, results

def plot_confusion_matrix(actual, predicted, labels, title):
    cm = confusion_matrix(actual, predicted, labels=labels)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_value = cm[i, j]
            if np.isnan(cell_value):
                text_value = ""
            else:
                text_value = str(cell_value)
                
            plt.text(j, i, text_value,
                     ha="center", va="center",
                     color="white" if cell_value > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Faktisk')
    plt.xlabel('Forudsagt')
    plt.show()

def main():
    terrain_model = load_terrain_model()
    if terrain_model is None:
        print("Kunne ikke indlæse terrænklassifikationsmodel. Afslutter.")
        return
    
    templates, template_names = load_crown_templates()
    if not templates:
        print("Kunne ikke indlæse crown templates. Afslutter.")
        return
    
    print("Evaluerer det samlede system på udvalgte samples...")
    stats, results = evaluate_on_sample(
        terrain_model, templates, template_names, 
        samples_per_terrain=10, visualize=True
    )
    
    actual_terrains = stats['actual_terrains']
    predicted_terrains = stats['predicted_terrains']
    actual_crowns = stats['actual_crowns']
    predicted_crowns = stats['predicted_crowns']

    try:
        all_terrain_labels = sorted(list(set(actual_terrains + predicted_terrains)))
        if len(all_terrain_labels) > 1:
            plot_confusion_matrix(
                actual_terrains, 
                predicted_terrains,
                all_terrain_labels,
                "Terrain Classification Confusion Matrix"
            )
    except Exception as e:
        print(f"Kunne ikke plotte terrain confusion matrix: {e}")
    
    try:
        all_crown_counts = sorted(list(set(actual_crowns + predicted_crowns)))
        if len(all_crown_counts) > 1:
            plot_confusion_matrix(
                actual_crowns, 
                predicted_crowns,
                all_crown_counts,
                "Crown Detection Confusion Matrix (Counts)"
            )
    except Exception as e:
        print(f"Kunne ikke plotte crown confusion matrix: {e}")

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import deque # Behøves egentlig ikke her, men bevares da den var i originalen

# Vigtigt: Sikr at 'old.crown_detector' er korrekt sti til den fil, der indeholder funktionen
# Da dette script *er* old/crown_detector.py, kan importen fjernes,
# og funktionen bruges direkte. Jeg fjerner importen og bevarer funktionen her.

def load_template(template_path):
    template = cv2.imread(template_path)
    if template is None:
        raise FileNotFoundError(f"Kunne ikke indlæse template fra {template_path}")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    return template_gray

def detect_crowns_in_tile(tile_image, templates, template_names, threshold=0.6, visualize=False):
    tile_gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB)
    tile_gray = cv2.cvtColor(tile_gray, cv2.COLOR_RGB2GRAY)
    
    best_matches = []
    
    for template, template_name in zip(templates, template_names):
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
            
            if distance > min(best_match['template_size']) // 2:
                non_overlapping_matches.append(match)
        
        best_matches = non_overlapping_matches

    # Returner antal detektioner og en liste af centroid koordinater (lokal til tile)
    crown_count = len(final_matches)
    centroids = [(match['position'][0] + match['template_size'][1] // 2, 
                  match['position'][1] + match['template_size'][0] // 2) 
                 for match in final_matches]

    if visualize and final_matches:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB))
        plt.title("Originalt tile")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        result_img = cv2.cvtColor(tile_image.copy(), cv2.COLOR_BGR2RGB)
        
        for match in final_matches:
            x, y = match['position']
            w, h = match['template_size'][1], match['template_size'][0]
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{match['template']} ({match['value']:.2f})"
            cv2.putText(result_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        plt.imshow(result_img)
        plt.title(f"Detekterede kroner: {crown_count}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Den oprindelige evaluate_crown_detection funktion returnerede kun count,
    # men classifieren kaldte en funktion der returnerede count OG centroids.
    # Vi ændrer denne funktion til at matche signaturen fra classifieren.
    return crown_count, centroids


def extract_crown_count_from_filename(filename):
    match = re.search(r'(\d+)crowns', filename)
    if match:
        return int(match.group(1))
    return 0

def evaluate_crown_detection(terrain_dir, templates, template_names, threshold=0.6, max_files=None, visualize=False):
    files = [f for f in os.listdir(terrain_dir) if f.endswith('.png')]
    
    if max_files is not None:
        files = files[:max_files]
    
    results = []
    for i, file in enumerate(files):
        file_path = os.path.join(terrain_dir, file)
        
        tile = cv2.imread(file_path)
        if tile is None:
            print(f"Kunne ikke indlæse {file_path}")
            continue
        
        actual_crowns = extract_crown_count_from_filename(file)
        
        # Kald detect_crowns_in_tile - vi bruger kun antallet her til evaluering
        detected_crowns, _ = detect_crowns_in_tile(
            tile, templates, template_names, threshold, 
            visualize=(visualize and i < 5)
        )
        
        results.append({
            'filename': file,
            'actual_crowns': actual_crowns,
            'detected_crowns': detected_crowns,
            'correct': actual_crowns == detected_crowns
        })
        
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total if total > 0 else 0
    
    true_positives = sum(1 for r in results if r['actual_crowns'] > 0 and r['detected_crowns'] > 0)
    false_positives = sum(1 for r in results if r['actual_crowns'] == 0 and r['detected_crowns'] > 0)
    false_negatives = sum(1 for r in results if r['actual_crowns'] > 0 and r['detected_crowns'] == 0)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    stats = {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print("\nEksempler på fejldetektioner:")
    errors = [r for r in results if not r['correct']]
    for i, error in enumerate(errors[:5]):
        print(f"Fil: {error['filename']}, Faktisk: {error['actual_crowns']}, Detekteret: {error['detected_crowns']}")
    
    if len(errors) > 5:
        print(f"...og {len(errors) - 5} mere.")
    
    return stats, results

def main():
    template_dir = "KingDominoDataset/Crown_Templates"
    template_paths = [
        os.path.join(template_dir, "Crown_up.png"),
        os.path.join(template_dir, "Crown_down.png"),
        os.path.join(template_dir, "Crown_left.png"),
        os.path.join(template_dir, "Crown_right.png")
    ]
    template_names = ["Up", "Down", "Left", "Right"]
    
    print("Indlæser templates...")
    templates = []
    for path in template_paths:
        try:
            template = load_template(path)
            templates.append(template)
            print(f"Indlæst template: {path}")
        except FileNotFoundError as e:
            print(f"Advarsel: {e}")
    
    if len(templates) != len(template_paths):
        print("Kunne ikke indlæse alle templates. Kontroller filstierne.")
        return
    
    terrain_categories_dir = "KingDominoDataset/TerrainCategories"
    terrain_types = ["Field", "Forest", "Grassland", "Lake", "Mine", "Swamp"]
    
    threshold = 0.6
    max_files = 50
    
    all_results = {}
    for terrain in terrain_types:
        terrain_dir = os.path.join(terrain_categories_dir, terrain)
        
        if not os.path.exists(terrain_dir):
            print(f"Advarsel: Mappen {terrain_dir} findes ikke.")
            continue
        
        print(f"\nEvaluerer kronedetektering for {terrain}...")
        stats, results = evaluate_crown_detection(
            terrain_dir, templates, template_names, 
            threshold, max_files, visualize=True
        )
        
        print(f"Resultater for {terrain}:")
        print(f"  Nøjagtighed: {stats['accuracy']:.2f}")
        print(f"  Precision: {stats['precision']:.2f}")
        print(f"  Recall: {stats['recall']:.2f}")
        print(f"  F1-score: {stats['f1_score']:.2f}")
        
        all_results[terrain] = {
            'stats': stats,
            'results': results
        }
    
    total_correct = sum(r['stats']['correct'] for r in all_results.values())
    total_files = sum(r['stats']['total'] for r in all_results.values())
    overall_accuracy = total_correct / total_files if total_files > 0 else 0
    
    print("\nSamlet statistik på tværs af alle terræntyper:")
    print(f"  Total antal tiles evalueret: {total_files}")
    print(f"  Korrekte detektioner: {total_correct}")
    print(f"  Samlet nøjagtighed: {overall_accuracy:.2f}")
    
    if False:
        print("\nEksperimenterer med forskellige threshold-værdier...")
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        accuracies = []
        
        for thresh in thresholds:
            print(f"\nTester threshold = {thresh}")
            
            terrain = "Forest"
            terrain_dir = os.path.join(terrain_categories_dir, terrain)
            
            stats, _ = evaluate_crown_detection(
                terrain_dir, templates, template_names, 
                thresh, max_files=20, visualize=False
            )
            
            accuracies.append(stats['accuracy'])
            print(f"  Nøjagtighed ved threshold {thresh}: {stats['accuracy']:.2f}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, accuracies, 'o-')
        plt.xlabel('Threshold værdi')
        plt.ylabel('Nøjagtighed')
        plt.title('Nøjagtighed vs. Threshold værdi')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
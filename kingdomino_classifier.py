import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # Bruges til farvekonvertering

# Importér TerrainClassifier og relevante funktioner fra model.py
# Dette er nødvendigt for at indlæse modellen og udtrække features
from model import TerrainClassifier, load_model, extract_features

# Standard sti til modellen
MODEL_FILE = "kingdomino_terrain_model.pkl"

# Standard sti til et eksempelbillede
DEFAULT_IMAGE_PATH = r"KingDominoDataset\KingDominoDataset\Cropped and perspective corrected boards\74.jpg"

def load_board_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Billedfilen {image_path} findes ikke.")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Kunne ikke indlæse billedet fra {image_path}.")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image_rgb

def divide_board_into_tiles(image, grid_size=5):
    height, width, _ = image.shape
    
    tile_height = height // grid_size
    tile_width = width // grid_size
    
    tiles = []
    for row in range(grid_size):
        tile_row = []
        for col in range(grid_size):
            y_start = row * tile_height
            x_start = col * tile_width
            tile = image[y_start:y_start + tile_height, x_start:x_start + tile_width]
            tile_row.append(tile)
        tiles.append(tile_row)
    
    return tiles

def classify_tiles(tiles, model):
    grid_size = len(tiles)
    terrain_results = np.empty((grid_size, grid_size), dtype=object)
    
    all_features = []
    tile_positions = []
    
    for row in range(grid_size):
        for col in range(grid_size):
            tile_features = extract_features(tiles[row][col])
            if not isinstance(tile_features, np.ndarray):
                tile_features = np.array(tile_features)
            all_features.append(tile_features)
            tile_positions.append((row, col))
    
    all_features = np.array(all_features)
    terrain_types = model.predict_terrain(all_features)
    
    for (row, col), terrain_type in zip(tile_positions, terrain_types):
        terrain_results[row, col] = terrain_type
    
    return terrain_results

def visualize_classification_results(original_image, tiles, terrain_results, output_path=None):
    grid_size = len(tiles)
    
    terrain_colors = {
        'Field': 'gold',
        'Forest': 'darkgreen',
        'Lake': 'lightblue',
        'Mine': 'saddlebrown',
        'Swamp': 'olive',
        'Grassland': 'limegreen',
        'Home': 'gray',
        'Unknown': 'pink'
    }
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original plade")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    classification_img = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    
    for row in range(grid_size):
        for col in range(grid_size):
            terrain = terrain_results[row, col]
            color = terrain_colors.get(terrain, 'white')
            rgb = [int(x * 255) for x in mcolors.to_rgb(color)]
            classification_img[row, col] = rgb
    
    plt.imshow(classification_img)
    
    for row in range(grid_size):
        for col in range(grid_size):
            plt.text(col, row, terrain_results[row, col],
                     ha="center", va="center",
                     color="white", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5))
    
    plt.title("Klassifikationsresultat")
    plt.axis('off')
    
    plt.tight_layout()
    
    plt.figure(figsize=(15, 15))
    for row in range(grid_size):
        for col in range(grid_size):
            plt.subplot(grid_size, grid_size, row * grid_size + col + 1)
            plt.imshow(tiles[row][col])
            plt.title(f"{terrain_results[row, col]}", fontsize=9)
            plt.axis('off')
    
    plt.suptitle("Detaljeret terrænklassifikation", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualisering gemt til {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Klassificer King Domino-plade ved hjælp af trænet model')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE_PATH,
                        help='Sti til billedet af spillepladen')
    parser.add_argument('--model', type=str, default=MODEL_FILE,
                        help='Sti til den gemte model')
    parser.add_argument('--output', type=str, default='classification_result.png',
                        help='Sti til at gemme visualisering')
    parser.add_argument('--grid_size', type=int, default=5,
                        help='Størrelse af grid (standard: 5)')
    
    args = parser.parse_args()
    
    try:
        print(f"Indlæser model fra {args.model}...")
        model = load_model(args.model)
        
        print(f"Indlæser pladebillede fra {args.image}...")
        board_image = load_board_image(args.image)
        
        print(f"Opdeler pladen i {args.grid_size}x{args.grid_size} tiles...")
        tiles = divide_board_into_tiles(board_image, args.grid_size)
        
        print("Klassificerer tiles...")
        terrain_results = classify_tiles(tiles, model)
        
        print("Visualiserer resultater...")
        visualize_classification_results(board_image, tiles, terrain_results, args.output)
        
        print("\nKlassifikationsresultat:")
        for row in range(len(terrain_results)):
            print(" ".join([f"{terrain_results[row, col]:<10}" for col in range(len(terrain_results[row]))]))
        
        print("\nFuldført! Klassifikationsresultatet er gemt til", args.output)
        
    except FileNotFoundError as e:
        print(f"Fejl: {e}")
    except ValueError as e:
        print(f"Fejl i billedbehandling: {e}")
    except Exception as e:
        print(f"En uventet fejl opstod: {e}")

if __name__ == "__main__":
    main()
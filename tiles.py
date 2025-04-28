import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_board_image(image_path):
    image = cv2.imread(image_path)
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

def visualize_tiles(tiles, grid_size=5):
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    for row in range(grid_size):
        for col in range(grid_size):
            axes[row, col].imshow(tiles[row][col])
            axes[row, col].set_title(f"Tile ({row}, {col})")
            axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()

def save_tiles(tiles, output_dir, board_name):
    os.makedirs(output_dir, exist_ok=True)
    for row in range(len(tiles)):
        for col in range(len(tiles[row])):
            tile = tiles[row][col]
            tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            filename = f"{board_name}_tile_{row}_{col}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, tile_bgr)
    print(f"Alle tiles gemt i {output_dir}")

def process_board_images(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"Input-mappen '{input_dir}' blev ikke fundet.")
        exit(1)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        print(f"Ingen billedfiler (.png, .jpg, .jpeg) fundet i '{input_dir}'.")
        return

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        board_name = os.path.splitext(image_file)[0]
        
        print(f"Behandler {board_name}...")
        
        board_image = load_board_image(image_path)
        tiles = divide_board_into_tiles(board_image)
        save_tiles(tiles, output_dir, board_name)
        
        print(f"Færdig med {board_name}\n")

if __name__ == "__main__":
    input_dir = "King Domino dataset/King Domino dataset/Cropped and perspective corrected boards"
    output_dir = "King Domino dataset/King Domino dataset/Extracted_Tiles"
    
    process_board_images(input_dir, output_dir)

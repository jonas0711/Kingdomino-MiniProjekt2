import json
import os
import shutil

def organize_tiles_by_terrain(labels_file, tiles_dir, output_base_dir):
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    terrain_types = set()
    
    for board in labels_data.values():
        for tile_info in board.values():
            terrain = tile_info["terrain"]
            terrain_types.add(terrain)
    
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    for terrain in terrain_types:
        terrain_dir = os.path.join(output_base_dir, terrain)
        if not os.path.exists(terrain_dir):
            os.makedirs(terrain_dir)
            print(f"Oprettet mappe: {terrain_dir}")
    
    copied_count = 0
    for board_id, board_data in labels_data.items():
        for tile_pos, tile_info in board_data.items():
            filename = tile_info["filename"]
            terrain = tile_info["terrain"]
            crowns = tile_info["crowns"]
            
            source_path = os.path.join(tiles_dir, filename)
            new_filename = f"{board_id}_tile_{tile_pos}_{terrain}_{crowns}crowns.png"
            target_path = os.path.join(output_base_dir, terrain, new_filename)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                copied_count += 1
                
                if copied_count % 100 == 0:
                    print(f"Kopieret {copied_count} filer...")
            else:
                print(f"Advarsel: Filen {source_path} findes ikke")
    
    print(f"\nAfsluttet! Organiseret {copied_count} billeder i {len(terrain_types)} terrænmapper.")
    print("Terræntyper: " + ", ".join(sorted(terrain_types)))

if __name__ == "__main__":
    labels_file = "Excel+JSON/tile_labels_mapping.json"
    tiles_dir = "King Domino dataset/King Domino dataset/Extracted_Tiles"
    output_base_dir = "KingDominoDataset/TerrainCategories"
    
    organize_tiles_by_terrain(labels_file, tiles_dir, output_base_dir)
import pandas as pd
import os
import re
import json
from collections import defaultdict

def extract_coordinates(coord_text):
    match = re.search(r'\((\d+),\s*(\d+)\)', coord_text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None

def extract_terrain_and_crowns(terrain_text):
    if terrain_text is None or not isinstance(terrain_text, str) or not terrain_text.strip():
        return ("Unknown", 0)
    
    match = re.search(r'(\D+)(\d*)', terrain_text)
    if match:
        terrain = match.group(1).strip()
        crowns = match.group(2)
        crown_count = int(crowns) if crowns else 0
        return (terrain, crown_count)
    return (terrain_text, 0)

def load_labels_from_excel(excel_file):
    if not os.path.exists(excel_file):
        print(f"Fejl: Filen {excel_file} blev ikke fundet.")
        return {}
    
    all_labels = {}
    xl = pd.ExcelFile(excel_file)
    
    for sheet_name in xl.sheet_names:
        if sheet_name == 'Sheet':
            continue
        
        board_data = pd.read_excel(excel_file, sheet_name=sheet_name)
        board_labels = {}
        
        for _, row in board_data.iterrows():
            if len(row) >= 2:
                coord_text = row.iloc[0]
                terrain_text = row.iloc[1]
                
                coords = extract_coordinates(coord_text)
                
                if coords:
                    terrain, crowns = extract_terrain_and_crowns(terrain_text)
                    board_labels[coords] = (terrain, crowns)
        
        all_labels[sheet_name] = board_labels
    
    return all_labels

def map_labels_to_extracted_tiles(labels_dict, tiles_dir, output_file):
    if not os.path.exists(tiles_dir):
        print(f"Fejl: Mappen {tiles_dir} blev ikke fundet.")
        return
    
    tile_mapping = defaultdict(dict)
    tiles_files = os.listdir(tiles_dir)
    
    for filename in tiles_files:
        parts = filename.split('_')
        
        if len(parts) >= 4 and parts[1] == "tile":
            board_name = parts[0]
            row = int(parts[2])
            col = int(parts[3].split('.')[0])
            
            if board_name in labels_dict:
                board_labels = labels_dict[board_name]
                coords = (col, row)
                
                if coords in board_labels:
                    terrain, crowns = board_labels[coords]
                    
                    tile_mapping[board_name][f"{row}_{col}"] = {
                        "filename": filename,
                        "terrain": terrain,
                        "crowns": crowns
                    }
                else:
                    print(f"Advarsel: Ingen label fundet for {board_name} tile ({row}, {col})")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tile_mapping, f, indent=2)
    
    print(f"Mapping gemt til {output_file}")
    
    return tile_mapping

def create_labeled_dataset(excel_file, tiles_dir, output_file='tile_labels_mapping.json'):
    labels_dict = load_labels_from_excel(excel_file)
    mapping = map_labels_to_extracted_tiles(labels_dict, tiles_dir, output_file)
    if mapping is None:
        print("Ingen mapping blev oprettet. Afslutter.")
        return
    print(f"Labels indlæst for {len(labels_dict)} brætter.")
    terrain_counts = defaultdict(int)
    crown_counts = defaultdict(int)
    for board_name, tiles in mapping.items():
        for tile_pos, tile_info in tiles.items():
            terrain = tile_info["terrain"]
            crowns = tile_info["crowns"]
            terrain_counts[terrain] += 1
            crown_counts[crowns] += 1
    print("\nTerrænfordelingsstatistik:")
    for terrain, count in sorted(terrain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {terrain}: {count} tiles")
    print("\nKronefordelingsstatistik:")
    for crowns, count in sorted(crown_counts.items()):
        print(f"  {crowns} kroner: {count} tiles")
    return mapping

if __name__ == "__main__":
    excel_file = "Excel+JSON/kingdomino_labels_fixed.xlsx"
    tiles_dir = "King Domino dataset/King Domino dataset/Extracted_Tiles"
    output_file = "Excel+JSON/tile_labels_mapping.json"
    mapping = create_labeled_dataset(excel_file, tiles_dir, output_file)
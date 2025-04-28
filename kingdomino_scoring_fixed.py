import pandas as pd
import numpy as np
import re
import os

def extract_coordinates(coord_text):
    match = re.search(r'\((\d+),\s*(\d+)\)', coord_text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None

def extract_terrain_crowns(terrain_text):
    if terrain_text is None or not isinstance(terrain_text, str) or not terrain_text.strip():
        return ("Unknown", 0)
    
    match = re.search(r'(\D+)(\d*)', terrain_text)
    if match:
        terrain = match.group(1).strip()
        crowns = match.group(2)
        crown_count = int(crowns) if crowns else 0
        return (terrain, crown_count)
    return (terrain_text, 0)

def find_connected_territories(board, x, y, terrain, visited, home_coords):
    if x < 0 or x >= 5 or y < 0 or y >= 5:
        return []
    
    if (x, y) in visited:
        return []
    
    if (x, y) not in board:
        return []
    
    # Home tile connects but doesn't count towards territory size
    if (x, y) == home_coords:
        visited.add((x, y))
        territory = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            territory.extend(find_connected_territories(board, x + dx, y + dy, terrain, visited, home_coords))
        return territory
    
    elif board[(x, y)][0] != terrain:
        return []
    
    visited.add((x, y))
    territory = [(x, y)]
    
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        territory.extend(find_connected_territories(board, x + dx, y + dy, terrain, visited, home_coords))
    
    return territory

def calculate_territory_score(board, territory):
    num_tiles = len(territory)
    total_crowns = sum(board[tile][1] for tile in territory)
    
    if total_crowns == 0:
        return 0
    
    return num_tiles * total_crowns

def calculate_board_score(board_data):
    board = {}
    home_coords = None
    
    for _, row in board_data.iterrows():
        if len(row) < 2: # Ensure we have enough columns
            continue
        coord_text = row.iloc[0]
        terrain_text = row.iloc[1]
        
        coords = extract_coordinates(coord_text)
        terrain, crowns = extract_terrain_crowns(terrain_text)
        
        if coords:
            board[coords] = (terrain, crowns)
            if terrain == 'Home':
                home_coords = coords
    
    if home_coords is None:
        home_coords = (2, 2)
        print("Advarsel: Ingen Home tile fundet. Antager position (2,2).")
    
    visited = set()
    
    if home_coords in board:
        visited.add(home_coords)

    territories = []
    
    for (x, y) in board:
        if (x, y) not in visited:
            terrain = board[(x, y)][0]
            
            if terrain in ['Home', 'Table', 'Unknown']:
                visited.add((x, y))
                continue
                
            territory = find_connected_territories(board, x, y, terrain, visited, home_coords)
            if territory:
                territories.append((terrain, territory))
    
    total_score = 0
    territory_details = []
    
    for terrain_type, territory in territories:
        crowns = sum(board[tile][1] for tile in territory)
        tile_count = len(territory)
        score = calculate_territory_score(board, territory)
        
        total_score += score
        
        territory_details.append({
            'terrain_type': terrain_type,
            'tiles': tile_count,
            'crowns': crowns,
            'score': score
        })
    
    return total_score, territory_details

def calculate_harmony_bonus(board_data):
    board_coords = set()
    
    for _, row in board_data.iterrows():
        if len(row) < 1: # Ensure at least 1 column
            continue
        coord_text = row.iloc[0]
        coords = extract_coordinates(coord_text)
        if coords:
            board_coords.add(coords)
    
    if len(board_coords) == 25:
        all_coords = {(x, y) for x in range(5) for y in range(5)}
        if board_coords == all_coords:
            return 5
    
    return 0

def main():
    excel_file = 'kingdomino_labels_fixed.xlsx'
    
    if not os.path.exists(excel_file):
        print(f"Fejl: Filen {excel_file} blev ikke fundet.")
        return
    
    print(f"Indlæser Excel-filen {excel_file}...")
    xl = pd.ExcelFile(excel_file)
    
    all_scores = {}
    
    for sheet_name in xl.sheet_names:
        if sheet_name == 'Sheet':
            continue
            
        print(f"Beregner score for bræt {sheet_name}...")
        
        try:
            board_data = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            if board_data.shape[1] < 2:
                print(f"Advarsel: Fane {sheet_name} har ikke kolonne B. Springer over.")
                continue
            
            score, territory_details = calculate_board_score(board_data)
            harmony_bonus = calculate_harmony_bonus(board_data)
            total_score = score + harmony_bonus
            
            all_scores[sheet_name] = {
                'base_score': score,
                'harmony_bonus': harmony_bonus,
                'total_score': total_score,
                'territory_details': territory_details
            }
            
        except Exception as e:
            print(f"Fejl ved beregning af score for bræt {sheet_name}: {e}")
            import traceback
            traceback.print_exc() # Print traceback for debugging errors in specific sheets
    
    scores_df = pd.DataFrame({
        'Board': list(all_scores.keys()),
        'Base Score': [all_scores[board]['base_score'] for board in all_scores],
        'Harmony Bonus': [all_scores[board]['harmony_bonus'] for board in all_scores],
        'Total Score': [all_scores[board]['total_score'] for board in all_scores]
    })
    
    scores_df = scores_df.sort_values('Total Score', ascending=False)
    
    output_file = 'kingdomino_final_scores_fixed.xlsx'
    print(f"\nGemmer resultater til {output_file}...")
    
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            scores_df.to_excel(writer, sheet_name='Summary', index=False)
            for board in all_scores:
                details = all_scores[board]['territory_details']
                if details:
                    details_df = pd.DataFrame(details)
                    details_df = details_df.sort_values('score', ascending=False)
                    details_df.to_excel(writer, sheet_name=f'Board {board}', index=False)
        print(f"Resultater gemt til {output_file}")
    except Exception as e:
        print(f"Fejl ved gemning af Excel-fil: {e}")


    print("\nTop 5 brætter med højeste score:")
    top_5_boards = scores_df.head(5)
    for _, row in top_5_boards.iterrows():
        board = row['Board']
        base_score = row['Base Score']
        harmony_bonus = row['Harmony Bonus']
        total_score = row['Total Score']
        
        bonus_text = f" (inkl. {harmony_bonus} bonus point for harmoni)" if harmony_bonus > 0 else ""
        print(f"Bræt {board}: {total_score} point{bonus_text}")
    
    print("\nDetaljer for top 5 brætter:")
    for board in top_5_boards['Board']:
        base_score = all_scores[board]['base_score']
        harmony_bonus = all_scores[board]['harmony_bonus']
        total_score = all_scores[board]['total_score']
        
        bonus_text = f" (inkl. {harmony_bonus} bonus point for harmoni)" if harmony_bonus > 0 else ""
        print(f"\nBræt {board} (Total score: {total_score} point{bonus_text})")
        
        if all_scores[board]['territory_details']:
            print("Territorier:")
            for t in sorted(all_scores[board]['territory_details'], key=lambda x: x['score'], reverse=True):
                crown_text = "krone" if t['crowns'] == 1 else "kroner"
                tile_text = "felt" if t['tiles'] == 1 else "felter"
                
                print(f"  - {t['terrain_type']}: {t['tiles']} {tile_text}, {t['crowns']} {crown_text} = {t['score']} point")
        else:
            print("Ingen territorier med kroner fundet.")


if __name__ == "__main__":
    main()
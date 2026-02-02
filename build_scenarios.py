import pandas as pd
import numpy as np
import os
import sys

TRACE_DIR = "." 
OUTPUT_DIR = "scenarios"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

FILES = {
    'game': 'trace_gaming.csv',
    'video': 'trace_video.csv',
    'social': 'trace_social.csv',
    'idle': 'trace_idle.csv',
    'cam': 'trace_camera.csv'
}

PROFILES = {
    "scenario_gamer": {
        "duration_hours": 10.0, 
        "temp_offset": 0.0,
        "mix": [
            ('game',   0.75, (180, 600)), 
            ('video',  0.15, (60, 300)),
            ('social', 0.05, (30, 60)),
            ('idle',   0.05, (0, 0)) 
        ]
    },
    
    "scenario_binge_watcher": {
        "duration_hours": 15.0, 
        "temp_offset": 0.0,
        "mix": [
            ('video',  0.80, (600, 600)), 
            ('social', 0.10, (60, 180)),
            ('idle',   0.10, (300, 600))
        ]
    },

    "scenario_scroller": {
        "duration_hours": 15.0, 
        "temp_offset": 0.0,
        "mix": [
            ('social', 0.65, (60, 300)),
            ('video',  0.15, (30, 120)),
            ('idle',   0.10, (60, 300)),
            ('cam',    0.10, (10, 30))
        ]
    },

    "scenario_office": {
        "duration_hours": 24.0, 
        "temp_offset": 0.0,
        "mix": [
            ('idle',   0.65, (600, 600)), 
            ('social', 0.20, (30, 180)),
            ('video',  0.10, (300, 600)),
            ('cam',    0.05, (10, 30))
        ]
    },

    "scenario_polar": { 
        "duration_hours": 24.0,
        "temp_offset": -25.0,
        "mix": [
            ('idle',   0.65, (600, 600)),
            ('social', 0.20, (30, 180)),
            ('video',  0.10, (300, 600)),
            ('cam',    0.05, (10, 30))
        ]
    }
}

def load_traces():
    library = {}
    for key, filename in FILES.items():
        if not os.path.exists(filename):
            continue
        try:
            df = pd.read_csv(filename)
            if len(df) == 0:
                continue
            cols = [c for c in ['cpu', 's_on', 'br', 'net_kbps', 'temp_c'] if c in df.columns]
            library[key] = df[cols].copy()
            print(f"  -> loaded {key}: {len(df)} rows")
        except Exception as e:
            print(f"  -> Error loading {filename}: {e}")
    return library

def build_scenario(name, config, library):
    print(f"\nGenerating scenario {name} ...")
    
    target_len = int(config['duration_hours'] * 3600)
    timeline = []
    current_len = 0
    
    valid_mix = [m for m in config['mix'] if m[0] in library]
    if not valid_mix:
        return

    keys = [m[0] for m in valid_mix]
    probs = [m[1] for m in valid_mix]
    ranges = {m[0]: m[2] for m in valid_mix}
    
    probs = np.array(probs)
    if probs.sum() == 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / probs.sum()
    
    while current_len < target_len:
        k = np.random.choice(keys, p=probs)
        src = library[k]
        total_rows = len(src)
        min_d, max_d = ranges[k]
        actual_max = min(max_d, total_rows)
        if min_d > actual_max: 
            min_d = actual_max
            
        dur = np.random.randint(min_d, actual_max + 1)
        max_start_idx = total_rows - dur
        
        if max_start_idx <= 0:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, max_start_idx + 1)

        chunk = src.iloc[start_idx : start_idx + dur].copy()
        
        if config['temp_offset'] != 0 and 'temp_c' in chunk.columns:
            chunk['temp_c'] = chunk['temp_c'] + config['temp_offset']
            
        timeline.append(chunk)
        current_len += len(chunk)
        
    if not timeline: return

    full_df = pd.concat(timeline, ignore_index=True)
    full_df = full_df.iloc[:target_len].copy()
    
    full_df.reset_index(drop=True, inplace=True)
    full_df.insert(0, 'time_s', full_df.index)
    
    save_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    full_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    libs = load_traces()
    if libs:
        for name, conf in PROFILES.items():
            build_scenario(name, conf, libs)
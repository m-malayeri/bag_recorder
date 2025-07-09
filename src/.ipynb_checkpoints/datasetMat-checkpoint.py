import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import os

def create_mat_files(input_txt_file, output_dir=".", pos_dist_thr=2):
    """
    Reads a text file with format "image_path x y theta place_id" and creates 3 .mat files
    for train/validation/test splits. Each split follows the same structure with 90% db, 10% query.
    
    Args:
        input_txt_file: Path to input text file
        output_dir: Directory to save output .mat files
        pos_dist_thr: Positive distance threshold (default: 2)
    """
    
    # Read the text file
    data = []
    with open(input_txt_file, 'r') as file:
        for line in file:
            if not line.strip() or line.startswith("image_path"):
                continue
            
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) >= 5:  # Ensure we have all required fields
                    image_path = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    theta = float(parts[3])
                    place_id = int(parts[4])
                    data.append([image_path, x, y, theta, place_id])
    
    if not data:
        raise ValueError("No valid data found in the input file")
    
    print(f"Total data points: {len(data)}")
    
    # Convert to numpy arrays for easier manipulation
    image_paths = [row[0] for row in data]
    coordinates = np.array([[row[1], row[2]] for row in data])  # x, y coordinates
    
    # First split: 70% train, 30% temp (validation + test)
    train_indices, temp_indices = train_test_split(
        range(len(data)), 
        train_size=0.7, 
        random_state=42
    )
    
    # Second split: 10% validation, 20% test from the remaining 30%
    # This gives us 10%/(10%+20%) = 1/3 for validation, 2/3 for test
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=1/3,  # 10% of total / 30% of temp = 1/3
        random_state=42
    )
    
    print(f"Train split: {len(train_indices)} samples ({len(train_indices)/len(data)*100:.1f}%)")
    print(f"Validation split: {len(val_indices)} samples ({len(val_indices)/len(data)*100:.1f}%)")
    print(f"Test split: {len(test_indices)} samples ({len(test_indices)/len(data)*100:.1f}%)")
    
    # Create splits dictionary
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    # Process each split
    for split_name, split_indices in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Extract data for this split
        split_image_paths = [image_paths[i] for i in split_indices]
        split_coordinates = coordinates[split_indices]
        
        # Within each split: 90% database, 10% queries
        db_indices, q_indices = train_test_split(
            range(len(split_indices)), 
            test_size=0.1, 
            random_state=42 + hash(split_name) % 1000  # Different seed for each split
        )
        
        # Extract database and query data for this split
        db_image_paths = [split_image_paths[i] for i in db_indices]
        q_image_paths = [split_image_paths[i] for i in q_indices]
        
        db_coordinates = split_coordinates[db_indices].T  # Transpose to 2xN
        q_coordinates = split_coordinates[q_indices].T    # Transpose to 2xM
        
        # Create the dbStruct dictionary
        db_struct = {
            'whichSet': split_name,
            'dbImageFns': np.array(db_image_paths, dtype=object).reshape(-1, 1),  # Nx1
            'utmDb': db_coordinates,  # 2xN array of [x; y] coordinates (transposed)
            'qImageFns': np.array(q_image_paths, dtype=object).reshape(-1, 1),   # Mx1
            'utmQ': q_coordinates,   # 2xM array of [x; y] coordinates (transposed)
            'numImages': len(db_image_paths),
            'numQueries': len(q_image_paths),
            'posDistThr': pos_dist_thr,
            'posDistSqThr': pos_dist_thr ** 2,
            'nonTrivPosDistSqThr': 10  # As in original
        }
        
        # Save to .mat file
        output_file = os.path.join(output_dir, f"layer_{split_name}.mat")
        sio.savemat(output_file, {'dbStruct': db_struct})
        
        print(f"Created {output_file}")
        print(f"  Database images: {len(db_image_paths)}")
        print(f"  Query images: {len(q_image_paths)}")
        print(f"  Total for {split_name}: {len(split_image_paths)}")

# Example usage
if __name__ == "__main__":
    # Example usage - replace with your actual file paths
    input_file = "/home/jovyan/data/VPR/LW/image_pose_place.txt"  # Replace with your input text file path
    output_directory = "/home/jovyan/data/VPR/LW/datasets"  # Directory to save the .mat files
    
    # Check if input file exists
    if os.path.exists(input_file):
        try:
            create_mat_files(input_file, output_directory)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Input file '{input_file}' not found. Please update the file path.")
        print("\nTo use this script:")
        print("1. Replace 'your_input_file.txt' with the path to your text file")
        print("2. The text file should have format: 'image_path x y theta place_id' per line")
        print("3. Run the script to generate 3 .mat files:")
        print("   - layer_train.mat (70% of data)")
        print("   - layer_val.mat (10% of data)")  
        print("   - layer_test.mat (20% of data)")
        print("4. Each file contains 90% database images and 10% query images")
        print("5. No overlap between train/validation/test splits")

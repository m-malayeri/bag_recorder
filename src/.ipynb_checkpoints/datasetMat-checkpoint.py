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

def verify_mat_files(output_dir="."):
    """
    Verify all three created .mat files by loading and displaying their structure.
    """
    splits = ['train', 'val', 'test']
    
    print("\n" + "="*50)
    print("VERIFICATION OF ALL SPLITS")
    print("="*50)
    
    total_db_images = 0
    total_query_images = 0
    
    for split_name in splits:
        mat_file_path = os.path.join(output_dir, f"layer_{split_name}.mat")
        
        if not os.path.exists(mat_file_path):
            print(f"File {mat_file_path} not found!")
            continue
            
        try:
            mat_data = sio.loadmat(mat_file_path)
            db_struct = mat_data['dbStruct']
            
            num_db = db_struct['numImages'][0][0]
            num_q = db_struct['numQueries'][0][0]
            
            total_db_images += num_db
            total_query_images += num_q
            
            print(f"\n{split_name.upper()} SPLIT ({mat_file_path}):")
            print(f"  whichSet: {db_struct['whichSet'][0]}")
            print(f"  numImages (db): {num_db}")
            print(f"  numQueries: {num_q}")
            print(f"  Total: {num_db + num_q}")
            print(f"  posDistThr: {db_struct['posDistThr'][0][0]}")
            print(f"  dbImageFns shape: {db_struct['dbImageFns'].shape}")
            print(f"  utmDb shape: {db_struct['utmDb'].shape}")
            print(f"  qImageFns shape: {db_struct['qImageFns'].shape}")
            print(f"  utmQ shape: {db_struct['utmQ'].shape}")
            
            # Show first few elements as examples
            if len(db_struct['dbImageFns']) > 0:
                print(f"  First db image: {db_struct['dbImageFns'][0, 0]}")
            if len(db_struct['qImageFns']) > 0:
                print(f"  First query image: {db_struct['qImageFns'][0, 0]}")
                
        except Exception as e:
            print(f"Error verifying {mat_file_path}: {e}")
    
    print(f"\nSUMMARY:")
    print(f"Total database images across all splits: {total_db_images}")
    print(f"Total query images across all splits: {total_query_images}")
    print(f"Grand total: {total_db_images + total_query_images}")

def check_no_overlap(output_dir="."):
    """
    Verify that there's no overlap between train/val/test splits.
    """
    splits = ['train', 'val', 'test']
    all_images = {}
    
    print("\n" + "="*50)
    print("CHECKING FOR OVERLAP BETWEEN SPLITS")
    print("="*50)
    
    for split_name in splits:
        mat_file_path = os.path.join(output_dir, f"layer_{split_name}.mat")
        
        if not os.path.exists(mat_file_path):
            continue
            
        try:
            mat_data = sio.loadmat(mat_file_path)
            db_struct = mat_data['dbStruct']
            
            # Collect all images from this split
            db_images = [str(img) for img in db_struct['dbImageFns'].flatten()]
            q_images = [str(img) for img in db_struct['qImageFns'].flatten()]
            
            all_images[split_name] = set(db_images + q_images)
            print(f"{split_name}: {len(all_images[split_name])} unique images")
            
        except Exception as e:
            print(f"Error reading {mat_file_path}: {e}")
    
    # Check for overlaps
    if len(all_images) == 3:
        train_val_overlap = all_images['train'] & all_images['val']
        train_test_overlap = all_images['train'] & all_images['test']
        val_test_overlap = all_images['val'] & all_images['test']
        
        print(f"\nOverlap analysis:")
        print(f"Train-Val overlap: {len(train_val_overlap)} images")
        print(f"Train-Test overlap: {len(train_test_overlap)} images")
        print(f"Val-Test overlap: {len(val_test_overlap)} images")
        
        if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
            print("✓ NO OVERLAP - Splits are properly separated!")
        else:
            print("⚠ WARNING: Overlaps detected between splits!")

# Example usage
if __name__ == "__main__":
    # Example usage - replace with your actual file paths
    input_file = "/home/jovyan/data/VPR/LW/image_pose_place.txt"  # Replace with your input text file path
    output_directory = "/home/jovyan/data/VPR/LW/datasets"  # Directory to save the .mat files
    
    # Check if input file exists
    if os.path.exists(input_file):
        try:
            create_mat_files(input_file, output_directory)
            verify_mat_files(output_directory)
            check_no_overlap(output_directory)
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

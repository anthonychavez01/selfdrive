import os

# Path to your folder
folder_path = r"C:\Users\Chavez Anthony\OneDrive\Desktop\chavez\Autopilot-TensorFlow-master\selfdrive\Video Dataset"
# List all files in the folder
files = os.listdir(folder_path)

# Sort to maintain order (important if your files have sequence)
files.sort()

# Loop through and rename
for i, filename in enumerate(files, start=1):
    # Extract the file extension (.jpg, .png, etc.)
    ext = os.path.splitext(filename)[1]
    
    # Create new filename like 1.jpg, 2.png, etc.
    new_name = f"{i}{ext}"  # 001, 002, 003... use {i} if you want 1, 2, 3
    
    # Build full paths
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    
    # Rename the file
    os.rename(src, dst)

print("✅ Files renamed successfully!")

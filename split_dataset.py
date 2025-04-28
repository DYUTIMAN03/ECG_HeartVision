import os
import shutil
import random

# Set paths
source_dir = 'data/images'
train_dir = 'data/train'
test_dir = 'data/test'

# Split ratio
train_ratio = 0.8  # 80% train, 20% test

# Create train/test dirs
for split_dir in [train_dir, test_dir]:
    for label in ['normal', 'abnormal']:
        os.makedirs(os.path.join(split_dir, label), exist_ok=True)

# Split function
def split_and_copy(label):
    src_path = os.path.join(source_dir, label)
    all_files = os.listdir(src_path)
    random.shuffle(all_files)

    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    for f in train_files:
        shutil.copy(os.path.join(src_path, f), os.path.join(train_dir, label, f))
    for f in test_files:
        shutil.copy(os.path.join(src_path, f), os.path.join(test_dir, label, f))

    print(f"✅ {label.capitalize()}: {len(train_files)} train, {len(test_files)} test")

# Run for both classes
split_and_copy('normal')
split_and_copy('abnormal')

print("✅ Dataset split complete!")

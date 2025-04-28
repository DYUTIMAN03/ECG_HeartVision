import os

normal_count = len(os.listdir('data/images/normal'))
abnormal_count = len(os.listdir('data/images/abnormal'))

print(f"Normal images: {normal_count}")
print(f"Abnormal images: {abnormal_count}")


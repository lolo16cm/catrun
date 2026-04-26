import cv2
import os

cat_name = input("Enter cat name: ")
os.makedirs(f'dataset/{cat_name}', exist_ok=True)

# Start count from existing images
existing = len([f for f in os.listdir(f'dataset/{cat_name}') if f.endswith('.jpg')])
count = existing
print(f"Found {existing} existing images, continuing from {count}")

pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Camera failed")
    exit()

print("✅ Camera opened!")
print("Press ENTER to capture, type 'q' + ENTER to quit")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    key = input(f"[{count} saved] Press ENTER to capture, q to quit: ")

    if key.lower() == 'q':
        break
    else:
        path = f'dataset/{cat_name}/{count}.jpg'
        cv2.imwrite(path, frame)
        count += 1
        print(f'✅ Saved: {path}')

cap.release()
print(f'Done! Total {count} images for {cat_name}')

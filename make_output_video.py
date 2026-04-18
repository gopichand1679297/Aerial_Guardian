import cv2
import os


base_folder = "outputs"
video_folder = "videos"

os.makedirs(video_folder, exist_ok=True)

for folder in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"\n🎬 Creating video for: {folder}")

    # Only JPG images
    images = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]

    print(f"Found {len(images)} images")

    if len(images) == 0:
        continue

    def extract_number(name):
        try:
            return int(name.split("_")[-1].split(".")[0])
        except:
            return 0

    images.sort(key=extract_number)

    first_frame_path = os.path.join(folder_path, images[0])
    first_frame = cv2.imread(first_frame_path)

    if first_frame is None:
        continue

    height, width = first_frame.shape[:2]

    output_video = os.path.join(video_folder, f"{folder}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 20, (width, height))

    frame_count = 0

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f" Skipping broken image: {img_name}")
            continue

        frame = cv2.resize(frame, (width, height))

        out.write(frame)
        frame_count += 1

    out.release()

    print(f"Video created: {output_video} ({frame_count} frames)")

print("\nAll videos created successfully!")
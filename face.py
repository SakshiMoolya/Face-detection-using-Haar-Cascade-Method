import cv2

image_choices = {
    "1": "P7.jpeg",
    "2": "P6.jpg",
    "3": "P3.jpg",
    "4": "P1.jpg",
}

print("Select an image to detect faces:")
for key, value in image_choices.items():
    print(f"{key}: {value}")

choice = input("Enter the number of your choice: ").strip()

image_path = image_choices.get(choice)
if not image_path:
    print("Invalid choice. Exiting.")
    exit()

img = cv2.imread(image_path)
if img is None:
    print("Could not read the image. Please check the file path.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load Haar cascades
frontal_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
profile_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")

# If the image has only one person
one_person = input("Is this a single-person photo? (y/n): ").strip().lower()

# Single-person face detection 
if one_person == 'y':
    faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
    
    if len(faces) == 0:
        print("No face detected.")
    elif len(faces) > 1:
        print("Multiple faces detected. Expected only one.")
    else:
        (x, y, w, h) = faces[0]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        print("Single face detected.")
# General multi-face detection 
else:
    faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
    
    flipped_gray = cv2.flip(gray, 1)
    flipped_profiles = profile_cascade.detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

    for (x, y, w, h) in flipped_profiles:
        x = gray.shape[1] - x - w
        profiles = list(profiles) + [(x, y, w, h)]

    all_faces = list(faces) + profiles

    def is_overlap(face1, face2, threshold=0.3):
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2
        area1 = w1 * h1
        area2 = w2 * h2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        return inter_area / float(min(area1, area2)) > threshold

    final_faces = []
    for face in all_faces:
        if not any(is_overlap(face, kept) for kept in final_faces):
            final_faces.append(face)

    for (x, y, w, h) in final_faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    print(f"{len(final_faces)} face(s) detected.")

# Show result
cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

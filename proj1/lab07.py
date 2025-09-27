from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['cv2_imshow', 'cv_imshow']

import cv2
from IPython import display
import PIL


def cv2_imshow(a):
  """A replacement for cv2.imshow() for use in Jupyter notebooks.

  Args:
    a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
      (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
      image.
  """
  a = a.clip(0, 255).astype('uint8')
  # cv2 stores colors as BGR; convert to RGB
  if a.ndim == 3:
    if a.shape[2] == 4:
      a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
    else:
      a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
  display.display(PIL.Image.fromarray(a))


cv_imshow = cv2_imshow


def clear_folder(folder_path):
    """Usuń wszystkie pliki w danym folderze."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # Usuń plik
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Usuń pusty folder (jeśli istnieje)
        except Exception as e:
            print(f"Nie udało się usunąć {file_path}. Błąd: {e}")

# ###################################################################

import cv2
import mediapipe as mp
import os
import json
import glob

# Inicjalizacja MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def extract_keypoints_from_image(image):
    """Ekstrakcja punktów kluczowych (keypoints) z obrazu"""
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
    return keypoints


def save_keypoints_to_json(keypoints, output_folder, index, output_format="json"):
    """Zapisz keypoints do pliku w formacie JSON lub CSV"""
    # Formatuj numer z zerami (np. 1 -> '0001', 10 -> '0010', itd.)
    filename = f"keypoints_{index:04d}.{output_format}"  # Numeracja 4-cyfrowa
    output_path = os.path.join(output_folder, filename)

    if output_format == "json":
        with open(output_path, 'w') as json_file:
            json.dump(keypoints, json_file, indent=4)
        print(f"Keypoints zapisane do: {output_path}")

    elif output_format == "csv":
        # Zapisz keypoints w formacie CSV (przykładowe dane)
        with open(output_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for point in keypoints:
                writer.writerow(point)  # Zakładając, że keypoints są listą punktów (x, y)
        print(f"Keypoints zapisane do: {output_path}")


def process_image_and_save_keypoints(image_path, output_folder, index, output_format="json"):
    """Wczytaj obraz, ekstraktuj keypoints i zapisz do pliku JSON lub CSV"""
    image = cv2.imread(image_path)

    if image is None:
        print(f"Nie udało się wczytać obrazu z {image_path}")
        return

    # Ekstrakcja keypoints z obrazu
    keypoints = extract_keypoints_from_image(image)

    # Zapis keypoints do pliku w odpowiednim formacie
    save_keypoints_to_json(keypoints, output_folder, index, output_format)


def load_images_from_folder(folder_path):
    """Wczytaj obrazy z folderu"""
    supported_formats = ['*.jpg', '*.png']
    image_files = []
    for fmt in supported_formats:
        image_files.extend(glob.glob(os.path.join(folder_path, fmt)))
    image_files.sort()  # Sortuj obrazy, aby zachować kolejność
    return image_files


def process_folder_and_save_keypoints(images_folder, output_folder, output_format="json"):
    """Przetwórz wszystkie obrazy w folderze i zapisz keypoints do plików JSON lub CSV"""
    image_files = load_images_from_folder(images_folder)

    for idx, image_path in enumerate(image_files, start=1):  # Numeracja zaczyna się od 1
        process_image_and_save_keypoints(image_path, output_folder, idx, output_format)


class KeypointDataLoader:
    def __init__(self, data_path, file_format="json"):
        self.data_path = data_path
        self.file_format = file_format
        self.keypoints = self.load_keypoints()

    def load_keypoints(self):
        """Wczytaj keypoints z pliku JSON"""
        keypoints = []
        if self.file_format == "json":
            with open(self.data_path, 'r') as jsonfile:
                keypoints = json.load(jsonfile)
        return keypoints

    def visualize_keypoints_on_image(self, image):
        """Wizualizuj keypoints na obrazie"""
        for keypoint in self.keypoints:
            x, y = int(keypoint["x"] * image.shape[1]), int(keypoint["y"] * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        return image


def main():
    # Ścieżki do folderów
    images_folder = "/content/images"  # Folder z obrazami
    output_folder = "/content/keypoints"  # Folder do zapisu plików JSON

    # Tworzymy folder na keypoints, jeśli nie istnieje
    os.makedirs(output_folder, exist_ok=True)
    clear_folder(output_folder)
    # Przetwarzanie folderu i zapisywanie keypoints do plików JSON
    process_folder_and_save_keypoints(images_folder, output_folder, output_format="json")

    image_files = load_images_from_folder(images_folder)
    json_files = sorted(
        glob.glob(os.path.join(output_folder, "*.json")))  # Posortuj pliki JSON, aby odpowiadały obrazom

    for image_path, json_path in zip(image_files, json_files):
        loader = KeypointDataLoader(json_path, file_format="json")

        # Wczytaj obraz i wizualizuj keypoints
        image = cv2.imread(image_path)
        loader = KeypointDataLoader(json_path, file_format="json")
        img_with_keypoints = loader.visualize_keypoints_on_image(image)
        cv2_imshow(img_with_keypoints)


if __name__ == '__main__':
    main()
# ###################################################################\

import random


def augment_keypoints(keypoints, max_shift=10, scale_range=(0.8, 1.2)):
    """Augmentacja danych szkieletowych: przesunięcie i skalowanie (z uwzględnieniem struktury z 'x', 'y', 'z', 'visibility')"""

    # Przesunięcie punktów w przestrzeni 2D o losowy wektor
    shift_x = random.uniform(-max_shift, max_shift)
    shift_y = random.uniform(-max_shift, max_shift)

    # Skalowanie punktów w przestrzeni 2D
    scale = random.uniform(scale_range[0], scale_range[1])

    augmented_keypoints = []

    for point in keypoints:
        # Załóżmy, że keypointy są słownikami z kluczami: 'x', 'y', 'z', 'visibility'
        x = point['x']
        y = point['y']
        z = point['z']
        visibility = point['visibility']

        # Przesunięcie i skalowanie
        scale = random.uniform(-0.08, 0.08)
        augmented_x = (x * (1 + scale))
        augmented_y = (y * (1 + scale))
        augmented_z = z * (1 + scale)  # Skalowanie z (jeśli jest potrzebne)

        # Tworzenie nowego keypointa z przekształconymi wartościami
        augmented_keypoints.append({
            'x': augmented_x,
            'y': augmented_y,
            'z': augmented_z,
            'visibility': visibility
        })

    return augmented_keypoints


def load_keypoints_from_json(json_path):
    """Wczytaj keypoints z pliku JSON"""
    with open(json_path, 'r') as file:
        keypoints = json.load(file)
    return keypoints


def save_keypoints_to_json(keypoints, output_folder, filename):
    """Zapisz zaugmentedowane keypoints do pliku JSON"""
    output_path = os.path.join(output_folder, filename)
    with open(output_path, 'w') as json_file:
        json.dump(keypoints, json_file, indent=4)
    print(f"Zapisano zaugmentedowane keypoints do: {output_path}")


def augment_all_keypoints_in_folder(input_folder, output_folder, max_shift=10, scale_range=(0.8, 1.2)):
    """Augmentacja wszystkich keypoints w folderze i zapisanie wyników"""
    # Wczytanie plików JSON
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    os.makedirs(output_folder, exist_ok=True)
    clear_folder(output_folder)
    for json_file in json_files:
        json_path = os.path.join(input_folder, json_file)

        # Wczytanie keypoints z pliku
        keypoints = load_keypoints_from_json(json_path)

        # Augmentacja keypoints
        augmented_keypoints = augment_keypoints(keypoints, max_shift, scale_range)

        # Zapis zaugmentedowanych keypoints do nowego pliku
        augmented_json_file = f"augmented_{json_file}"
        save_keypoints_to_json(augmented_keypoints, output_folder, augmented_json_file)

# ###################################################################

# Ścieżki do folderów
input_folder = "/content/keypoints"  # Folder z oryginalnymi plikami JSON
output_folder = "/content/augmented_keypoints"  # Folder, gdzie zapiszemy zaugmentedowane keypoints

# Augmentacja z przesunięciem (max 10 pikseli) i skalowaniem w zakresie (0.8, 1.2)
augment_all_keypoints_in_folder(input_folder, output_folder, max_shift=10, scale_range=(0.8, 1.2))

# ###################################################################
import cv2
import glob
import os
from google.colab.patches import cv2_imshow

images_folder = "/content/images"  # Folder z obrazami
keypoints_folder = "/content/keypoints"  # Folder z plikami JSON z keypoints
augmented_folder = "/content/augmented_keypoints"  # Folder z plikami JSON z augmentowanymi keypoints

image_files = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))  # Wczytanie obrazów
keypoints_files = sorted(glob.glob(os.path.join(keypoints_folder, "*.json")))  # Wczytanie plików z keypoints
augmented_files = sorted(
    glob.glob(os.path.join(augmented_folder, "*.json")))  # Wczytanie plików z augmentowanymi keypoints

for image_path, keypoints_path, augmented_path in zip(image_files, keypoints_files, augmented_files):
    # Wczytaj obraz
    image1 = cv2.imread(image_path)
    image2 = cv2.imread(image_path)

    # Wczytaj keypoints (oryginalne)
    original_loader = KeypointDataLoader(keypoints_path, file_format="json")
    original_keypoints = original_loader.keypoints

    # Wczytaj keypoints po augmentacji
    augmented_loader = KeypointDataLoader(augmented_path, file_format="json")
    augmented_keypoints = augmented_loader.keypoints

    # Wizualizacja keypoints na obrazie oryginalnym
    original_image_with_keypoints = original_loader.visualize_keypoints_on_image(image1)

    # Wizualizacja keypoints na obrazie po augmentacji
    augmented_loader.keypoints = augmented_keypoints  # Zaktualizuj keypoints w loaderze augmentacji
    augmented_image_with_keypoints = augmented_loader.visualize_keypoints_on_image(image2)

    # Połączenie obu obrazów obok siebie
    combined_image = cv2.hconcat([original_image_with_keypoints, augmented_image_with_keypoints])

    # Wyświetlenie połączonego obrazu
    cv2_imshow(combined_image)

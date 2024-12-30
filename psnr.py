import os
import math
import cv2
import pydicom
import numpy as np
import pandas as pd
from scipy.signal import wiener
from tqdm import tqdm
import matplotlib.pyplot as plt

def psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Oblicza PSNR (Peak Signal-to-Noise Ratio) między oryginalnym a przetworzonym obrazem.

    Parametry:
        original (np.ndarray): Oryginalny obraz w skali szarości.
        processed (np.ndarray): Przetworzony obraz w skali szarości.

    Zwraca:
        float: Wartość PSNR. Jeśli obrazy są identyczne (MSE=0), zwraca 100.
    """
    mse = np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def add_gaussian_noise(image: np.ndarray, variance: float = 0.01) -> np.ndarray:
    """
    Dodaje szum Gaussowski do obrazu w skali szarości.

    Parametry:
        image (np.ndarray): Obraz w skali szarości (uint8).
        variance (float): Wariancja szumu Gaussowskiego (np. 0.01).

    Zwraca:
        np.ndarray: Obraz z dodanym szumem Gaussowskim.
    """
    row, col = image.shape
    sigma = (variance ** 0.5) * 255
    gauss = np.random.normal(0, sigma, (row, col))
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.01) -> np.ndarray:
    """
    Dodaje szum typu Salt & Pepper do obrazu w skali szarości.

    Parametry:
        image (np.ndarray): Obraz w skali szarości (uint8).
        amount (float): Proporcja pikseli do zaszumienia (np. 0.01).

    Zwraca:
        np.ndarray: Obraz z dodanym szumem Salt & Pepper.
    """
    noisy = image.copy()
    num_salt = int(np.ceil(amount * image.size * 0.5))
    num_pepper = int(np.ceil(amount * image.size * 0.5))

    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]

    noisy[coords_salt[0], coords_salt[1]] = 255
    noisy[coords_pepper[0], coords_pepper[1]] = 0
    return noisy

def apply_median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Zastosowanie filtra medianowego.

    Parametry:
        image (np.ndarray): Obraz w skali szarości (uint8).
        ksize (int): Rozmiar jądra filtra medianowego (np. 3).

    Zwraca:
        np.ndarray: Obraz po filtracji medianowej.
    """
    return cv2.medianBlur(image, ksize)

def apply_average_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Zastosowanie filtra uśredniającego (blur).

    Parametry:
        image (np.ndarray): Obraz w skali szarości (uint8).
        ksize (int): Rozmiar jądra filtra (np. 3).

    Zwraca:
        np.ndarray: Obraz po filtracji uśredniającej.
    """
    return cv2.blur(image, (ksize, ksize))

def apply_gaussian_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Zastosowanie filtra Gaussowskiego.

    Parametry:
        image (np.ndarray): Obraz w skali szarości (uint8).
        ksize (int): Rozmiar jądra filtra (np. 3).

    Zwraca:
        np.ndarray: Obraz po filtracji Gaussowskiej.
    """
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_wiener_filter(image: np.ndarray) -> np.ndarray:
    """
    Zastosowanie filtra Wienera (scipy.signal.wiener).

    Parametry:
        image (np.ndarray): Obraz w skali szarości (uint8).

    Zwraca:
        np.ndarray: Obraz po filtracji Wienera.
    """
    wienered = wiener(image, mysize=(3, 3))
    wienered = np.clip(wienered, 0, 255).astype(np.uint8)
    return wienered

def apply_weighted_median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Przykładowa implementacja filtra ważonej mediany.

    Parametry:
        image (np.ndarray): Obraz w skali szarości (uint8).
        ksize (int): Rozmiar jądra filtra (np. 3).

    Zwraca:
        np.ndarray: Obraz po filtracji ważonej medianie.
    """
    weights = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]], dtype=np.float32)
    pad = ksize // 2
    rows, cols = image.shape
    filtered = np.zeros_like(image)

    for r in range(pad, rows - pad):
        for c in range(pad, cols - pad):
            patch = image[r - pad:r + pad + 1, c - pad:c + pad + 1].flatten()
            w = weights.flatten()
            idx_sorted = np.argsort(patch)
            patch_sorted = patch[idx_sorted]
            w_sorted = w[idx_sorted]

            cumsum = np.cumsum(w_sorted)
            half = cumsum[-1] / 2.0
            median_idx = np.where(cumsum >= half)[0][0]
            filtered[r, c] = patch_sorted[median_idx]
    return filtered

def apply_clahe(image: np.ndarray, clipLimit: float = 2.0, tileGridSize: tuple = (8, 8)) -> np.ndarray:
    """
    Zastosowanie CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parametry:
        image (np.ndarray): Obraz w skali szarości (uint8).
        clipLimit (float): Maksymalny kontrast dla CLAHE.
        tileGridSize (tuple): Wielkość siatki dla CLAHE.

    Zwraca:
        np.ndarray: Obraz po zastosowaniu CLAHE.
    """
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(image)

def display_median_clahe(
    original: np.ndarray,
    median_filtered: np.ndarray,
    clahe_filtered: np.ndarray,
    patient_id: str,
    dicom_name: str,
    output_images_path: str
) -> None:
    """
    Zapisuje porównanie trzech obrazów (oryginał, filtr medianowy, filtr medianowy + CLAHE)
    w formie pliku PNG.

    Parametry:
        original (np.ndarray): Oryginalny obraz w skali szarości.
        median_filtered (np.ndarray): Obraz po filtrze medianowym.
        clahe_filtered (np.ndarray): Obraz po filtrze medianowym i CLAHE.
        patient_id (str): Identyfikator pacjenta.
        dicom_name (str): Nazwa pliku DICOM (bez ścieżki).
        output_images_path (str): Folder, w którym zostaną zapisane wyniki.
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Oryginał')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(median_filtered, cmap='gray')
    plt.title('Filtr medianowy')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(clahe_filtered, cmap='gray')
    plt.title('Filtr medianowy + CLAHE')
    plt.axis('off')

    plt.tight_layout()
    image_filename = f"{patient_id}_{dicom_name}_Median_CLAHE.png"
    plt.savefig(os.path.join(output_images_path, image_filename))
    plt.close()

def calculate_black_ratio(dicom_path: str, threshold: int = 10) -> float:
    """
    Oblicza procent pikseli o jasności < threshold w obrazie DICOM.

    Parametry:
        dicom_path (str): Ścieżka do pliku DICOM.
        threshold (int): Próg dla "czarnych" pikseli (domyślnie 10).

    Zwraca:
        float: Stosunek liczby czarnych pikseli do całkowitej liczby pikseli (w zakresie [0,1]).
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        black_pixels = np.sum(image < threshold)
        total_pixels = image.size
        return black_pixels / total_pixels
    except Exception as e:
        print(f"Nie udało się obliczyć black ratio dla {dicom_path} z powodu: {e}")
        return 1.0

def find_dicom_with_least_black(patient_path: str) -> str:
    """
    Znajduje w podanym folderze (i podfolderach) plik DICOM z najmniejszym odsetkiem czarnych pikseli.

    Parametry:
        patient_path (str): Ścieżka do folderu pacjenta.

    Zwraca:
        str lub None: Ścieżka do wybranego pliku DICOM (lub None, jeśli nie znaleziono).
    """
    dicom_files = []
    for root, dirs, files in os.walk(patient_path):
        for f in files:
            if f.endswith(".dcm"):
                dicom_files.append(os.path.join(root, f))

    if not dicom_files:
        return None

    black_ratios = {}
    for dcm_path in dicom_files:
        black_ratios[dcm_path] = calculate_black_ratio(dcm_path)

    best_dcm = min(black_ratios, key=black_ratios.get)
    return best_dcm


def main():
    """
    Główna funkcja skryptu:
    1. Ustawia ścieżki do folderów i plików wyjściowych.
    2. Przechodzi po folderach pacjentów i znajduje plik DICOM z najmniejszą ilością czarnych pikseli.
    3. Dla znalezionego pliku:
       - Normalizuje obraz do 0-255.
       - Dodaje różne rodzaje szumu (Gaussian, Salt & Pepper) o różnych wariancjach.
       - Filtruje (medianowy, Wiener, Average, Gaussian, Weighted Median) + CLAHE.
       - Oblicza PSNR i gromadzi wyniki w słowniku.
    4. Zapisuje wyniki do pliku Excel (psnr_results_mammo.xlsx).
    5. Generuje i zapisuje obrazy poglądowe (dla jednego wybranego pacjenta).
    6. Oblicza średnie PSNR w zależności od filtra, szumu i wariancji i dopisuje to w arkuszu 'PSNR_Average'.
    """

    base_path = r"C:\Users\a\Desktop\mammografia\manifest-1734734641099\CBIS-DDSM"
    psnr_output_excel = r"C:\Users\a\Desktop\mammografia\psnr_results_mammo.xlsx"
    output_images_path = r"C:\Users\a\Desktop\mammografia\Output_Images"
    os.makedirs(output_images_path, exist_ok=True)

    all_patients = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]

    results = {
        "Patient ID": [],
        "DICOM File": [],
        "Filter + CLAHE": [],
        "Noise Type": [],
        "Noise Variance": [],
        "PSNR": []
    }

    noise_variances = [0.001, 0.01, 0.1]
    noise_types = ["Gaussian", "Salt_Pepper"]

    target_patient_index = 1

    for idx, patient_id in enumerate(tqdm(all_patients, desc="Przetwarzanie pacjentów")):
        patient_path = os.path.join(base_path, patient_id)

        best_dcm_path = find_dicom_with_least_black(patient_path)
        if not best_dcm_path:
            print(f"[{patient_id}] Brak plików DICOM lub nie można wczytać. Pomijanie...")
            continue

        try:
            ds = pydicom.dcmread(best_dcm_path)
            original = ds.pixel_array

            if original.dtype != np.uint8:
                original = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            dcm_basename = os.path.basename(best_dcm_path)

            for n_type in noise_types:
                for var in noise_variances:
                    if n_type == "Gaussian":
                        noisy = add_gaussian_noise(original, variance=var)
                    else:
                        noisy = add_salt_pepper_noise(original, amount=var)

                    median_img = apply_median_filter(noisy)
                    median_clahe = apply_clahe(median_img)
                    psnr_after = psnr(original, median_clahe)

                    results["Patient ID"].append(patient_id)
                    results["DICOM File"].append(dcm_basename)
                    results["Filter + CLAHE"].append("Median Filter + CLAHE")
                    results["Noise Type"].append(n_type)
                    results["Noise Variance"].append(var)
                    results["PSNR"].append(psnr_after)

                    if idx == target_patient_index:
                        display_median_clahe(
                            original,
                            median_img,
                            median_clahe,
                            patient_id,
                            dcm_basename,
                            output_images_path
                        )

                    wiener_img = apply_wiener_filter(noisy)
                    wiener_clahe = apply_clahe(wiener_img)
                    psnr_after = psnr(original, wiener_clahe)

                    results["Patient ID"].append(patient_id)
                    results["DICOM File"].append(dcm_basename)
                    results["Filter + CLAHE"].append("Wiener Filter + CLAHE")
                    results["Noise Type"].append(n_type)
                    results["Noise Variance"].append(var)
                    results["PSNR"].append(psnr_after)

                    avg_img = apply_average_filter(noisy)
                    avg_clahe = apply_clahe(avg_img)
                    psnr_after = psnr(original, avg_clahe)

                    results["Patient ID"].append(patient_id)
                    results["DICOM File"].append(dcm_basename)
                    results["Filter + CLAHE"].append("Average Filter + CLAHE")
                    results["Noise Type"].append(n_type)
                    results["Noise Variance"].append(var)
                    results["PSNR"].append(psnr_after)

                    gauss_img = apply_gaussian_filter(noisy)
                    gauss_clahe = apply_clahe(gauss_img)
                    psnr_after = psnr(original, gauss_clahe)

                    results["Patient ID"].append(patient_id)
                    results["DICOM File"].append(dcm_basename)
                    results["Filter + CLAHE"].append("Gaussian Filter + CLAHE")
                    results["Noise Type"].append(n_type)
                    results["Noise Variance"].append(var)
                    results["PSNR"].append(psnr_after)

                    wmedian_img = apply_weighted_median_filter(noisy)
                    wmedian_clahe = apply_clahe(wmedian_img)
                    psnr_after = psnr(original, wmedian_clahe)

                    results["Patient ID"].append(patient_id)
                    results["DICOM File"].append(dcm_basename)
                    results["Filter + CLAHE"].append("Weighted Median Filter + CLAHE")
                    results["Noise Type"].append(n_type)
                    results["Noise Variance"].append(var)
                    results["PSNR"].append(psnr_after)

        except Exception as ex:
            print(f"[{patient_id}] Błąd podczas przetwarzania pliku {best_dcm_path}: {ex}")

    print("\nZakończono zbieranie wyników, tworzenie tabeli...\n")

    df_results = pd.DataFrame(results)
    df_results.sort_values(
        by=["Patient ID", "DICOM File", "Filter + CLAHE", "Noise Type", "Noise Variance"],
        inplace=True
    )

    df_results.to_excel(psnr_output_excel, index=False)
    print(f"Wyniki PSNR zapisano w: {psnr_output_excel}")

    mean_df = df_results.groupby(["Filter + CLAHE", "Noise Type", "Noise Variance"]).agg({
        "PSNR": "mean"
    }).reset_index()

    with pd.ExcelWriter(psnr_output_excel, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        mean_df.to_excel(writer, sheet_name="PSNR_Average", index=False)

    print("Średnie wartości PSNR obliczono i zapisano w arkuszu 'PSNR_Average'.\n")
    print("Gotowe!")

if __name__ == "__main__":
    main()

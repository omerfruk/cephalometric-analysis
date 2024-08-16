import os
import cv2
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

image_folder = 'RawImage/TrainingData/'
coords_folder = 'AnnotationsByMD/400_senior/'


def load_data(image_folder, coords_folder):
    data = []  # Tüm veriyi tutacak bir liste

    # Görüntülerin ve koordinat dosyalarının isimlerini alalım
    image_files = os.listdir(image_folder)
    for image_file in image_files:
        # Görüntünün tam yolu
        image_path = os.path.join(image_folder, image_file)

        # Aynı isimli koordinat dosyasını bulalım
        coord_file = os.path.splitext(image_file)[0] + '.txt'
        coord_path = os.path.join(coords_folder, coord_file)

        # Görüntüyü yükleme
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Koordinatları yükleme
        coords_df = pd.read_csv(coord_path, sep=' ', header=None, dtype=str)
        coords = coords_df[0].str.split(',', expand=True).astype('float32').values

        # Veri yapısına ekleme
        data.append({
            'image': image,
            'coords': coords
        })
    output_csv_path = os.path.join(coords_folder, f"{os.path.splitext(image_file)[0]}_processed.csv")
    pd.DataFrame(coords, columns=['x', 'y']).to_csv(output_csv_path, index=False)

    return data


# Basit bir CNN modelini oluşturma fonksiyonu
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5 * 2)  # 5 nokta x 2 koordinat (x, y) = 10 çıkış
    ])

    model.compile(optimizer=Adam(),
                  loss=MeanSquaredError(),
                  metrics=['accuracy'])

    return model


# Veriyi uygun formata getirme fonksiyonu
def preprocess_data(data):
    images = np.array([item['image'] for item in data])
    images = images[..., np.newaxis]  # Kanal boyutunu ekleme (grayscale)
    coords = np.array([item['coords'].flatten()[:10] for item in data])
    return images, coords


# Veriyi karıştırma ve bölme fonksiyonu
def split_data(data):
    random.shuffle(data)

    total_size = len(data)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


# Ana işlem
def main():
    # Veriyi yükleme
    data = load_data(image_folder, coords_folder)

    # Veriyi bölme
    train_data, val_data, test_data = split_data(data)

    # Veriyi uygun formata getirme
    train_images, train_coords = preprocess_data(train_data)
    val_images, val_coords = preprocess_data(val_data)
    test_images, test_coords = preprocess_data(test_data)

    # Modeli oluşturma
    input_shape = (1935, 2400, 1)  # Görüntü boyutu (yükseklik, genişlik, kanal sayısı)
    model = create_model(input_shape)

    # Modeli eğitme
    history = model.fit(
        train_images, train_coords,
        epochs=10,  # Eğitim döngüleri sayısı
        batch_size=16,
        validation_data=(val_images, val_coords)
    )

    # Eğitim ve doğrulama kaybını çizme
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Modeli test etme
    test_loss = model.evaluate(test_images, test_coords)
    print(f'Test Loss: {test_loss}')


if __name__ == "__main__":
    main()

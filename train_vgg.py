import os
import glob
import numpy as np
from skimage import io, transform
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical

# Параметры
path = "/Users/mihail/Desktop/новогодний/vgg16/picture/"
w, h, c = 224, 224, 3

def read_img(path):
    cate = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    imgs, labels = [], []
    for idx, folder in enumerate(cate):
        for im in glob.glob(os.path.join(folder, '*.jpg')):
            print(f'Reading the image: {im}')
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

# Загрузка данных
data, label = read_img(path)

# Перемешивание данных
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data, label = data[arr], label[arr]

# Разделение на тренировочные и валидационные данные
ratio = 0.8
s = int(num_example * ratio)
x_train, y_train = data[:s], label[:s]
x_val, y_val = data[s:], label[s:]

# Преобразование меток в one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

def build_network(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model

def train_network(model, x_train, y_train, x_val, y_val, batch_size, epochs, save_path):
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
    )
    model.save(save_path)
    return history

def main():
    input_shape = (w, h, c)
    num_classes = 2
    batch_size = 12
    epochs = 50
    save_path = "vgg16_model.h5"

    model = build_network(input_shape, num_classes)
    train_network(model, x_train, y_train, x_val, y_val, batch_size, epochs, save_path)

if __name__ == "__main__":
    main()






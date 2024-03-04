import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# Generar un timestamp único
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Rutas de los conjuntos de datos
train_dir = '../data/train'  # Ajusta según la ubicación real de la carpeta data
test_dir = '../data/test'    # Ajusta según la ubicación real de la carpeta data

# Configuración del generador de datos para el conjunto de entrenamiento
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Configuración del generador de datos para el conjunto de prueba
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Construir el modelo CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo y obtener el historial
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

# Visualizar y guardar la precisión con timestamp único
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Guardar la gráfica de precisión con timestamp único
accuracy_plot_path = f'../models/accuracy_plot_{timestamp}.jpg'
plt.savefig(accuracy_plot_path)
plt.show()

# Guardar el historial en un archivo con pickle con timestamp único
history_file_path = f'../models/history_{timestamp}.pkl'
with open(history_file_path, 'wb') as file:
    pickle.dump(history.history, file)

# Guardar el modelo en el formato nativo de Keras (.keras) con timestamp único
model_path_keras = f'../models/eye_detection_model_{timestamp}.keras'
model.save(model_path_keras)


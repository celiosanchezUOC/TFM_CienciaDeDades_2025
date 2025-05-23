import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS_INITIAL = 5
EPOCHS_FINE = 10
MODEL_PATH = "best_model.h5"

# Rutas a los datos
train_dir = 'Dataset/train'
val_dir = 'Dataset/val'
test_dir = 'Dataset/test'

# Preprocesamiento
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)

# Ruta local a los pesos preentrenados
local_weights = 'c:/CELIO/ModelosEntrenamiento/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Modelo base sin pesos
base_model = InceptionResNetV2(
    weights=None,
    include_top=False,
    input_shape=(299, 299, 3)
)

# Cargar pesos localmente
base_model.load_weights(local_weights)

# Congelar capas base
for layer in base_model.layers:
    layer.trainable = False

# Añadir capas superiores
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compilación
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='AUC'), tf.keras.metrics.Recall(name='Recall')])

# Callbacks
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entrenamiento inicial
print("🔧 Entrenando primeras capas...")
history_initial = model.fit(train_generator,
                            validation_data=val_generator,
                            epochs=EPOCHS_INITIAL,
                            callbacks=[checkpoint, early_stop])

# Fine-tuning
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='AUC'), tf.keras.metrics.Recall(name='Recall')])

print("🔧 Fine-tuning...")
history_fine = model.fit(train_generator,
                         validation_data=val_generator,
                         epochs=EPOCHS_FINE,
                         callbacks=[checkpoint, early_stop])

# Cargar el mejor modelo
print(f"\n✅ Cargando mejor modelo desde {MODEL_PATH}")
model.load_weights(MODEL_PATH)

# Evaluación final
print("\n🔬 Evaluando en el conjunto TEST...")
test_generator.reset()
preds = model.predict(test_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes
target_names = list(test_generator.class_indices.keys())

print("\n📊 Reporte de clasificación (TEST):")
print(classification_report(y_true, y_pred, target_names=target_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=target_names, yticklabels=target_names, cmap="Blues")
plt.title("Matriz de Confusión - TEST")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# Guardar modelo final completo
model.save('inceptionresnetv2_finetuned_model_final.h5')
print("✅ Modelo completo guardado en 'inceptionresnetv2_finetuned_model_final.h5'")

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_true, preds[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - TEST')
plt.legend(loc='lower right')
plt.show()

# Función para graficar historia
def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acc Train')
    plt.plot(history.history['val_accuracy'], label='Acc Val')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss Train')
    plt.plot(history.history['val_loss'], label='Loss Val')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Graficar historia
plot_history(history_initial)
plot_history(history_fine)

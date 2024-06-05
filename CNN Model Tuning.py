import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.4)
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.applications.vgg16 import VGG16
from sklearn import decomposition
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from sklearn.metrics import accuracy_score

#Załadowanie wygenerowanych wcześniej przekształconych danych (obrazków)
#Upload previously generated transformed data (images).
train_images = np.load("TransformedData/train_images.npy")
train_labels = np.load("TransformedData/train_labels.npy")
test_images = np.load("TransformedData/test_images.npy")
test_labels = np.load("TransformedData/test_labels.npy")
class_names = np.load("TransformedData/class_names.npy")
class_labels = np.load("TransformedData/class_labels.npy", allow_pickle=True).item()


#Wczytanie wytrenowanego i zapisanego modelu (nie ma potrzeby tylko przykład jak to zrobić)
#Loading the trained and saved model (no need just an example of how you can do it)
saved_model_path = "Models/TrainedModel_NS- 0.76.h5"
model_76 = tf.keras.models.load_model(saved_model_path)

#Dostrajanie / ulepszanie modelu na podstawie modelu VGG16 z konkursu ImageNet
#Tuning / improving the model based on the VGG16 model from the ImageNet competition
model = VGG16(weights="imagenet", include_top=False)

#include_top=False oznacza, że chcemy załadować tylko konwolucyjne warstwy modelu, dzięki czemu możemy użyć tego modelu jako ekstraktora cech i dostosować go do naszego własnego problemu klasyfikacyjnego
#include_top=False means that we want to load only the convolutional layers of the model, so we can use this model as a feature extractor and customize it for our own classification problem


train_features = model.predict(train_images)
test_features = model.predict(test_images)

#Wizualizacja cech poprzez PCA
#Visualization of features through PCA
n_train, x, y, z = train_features.shape
n_test, x, y, z = test_features.shape
numFeatures = x * y * z

pca = decomposition.PCA(n_components=2)

X = train_features.reshape((n_train, numFeatures))
pca.fit(X)

C = pca.transform(X)
C1 = C[:, 0]
C2 = C[:, 1]

plt.subplots(figsize = (10, 10))
for i, class_name in enumerate(class_names):
    plt.scatter(C1[train_labels == i][:1000], C2[train_labels == i][:1000], label = class_name, alpha = 0.4)
plt.legend()
plt.title("PCA")
plt.show()

#Załadowanie wstępnie wytrenowanego modelu VGG16 bez warstw górnych
#Loading a pre-trained VGG16 model without top layers
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))


#Dodanie własnych warstw gęstych do załadowanego modelu - dodajemy identyczne jak w naszym modelu
#Add your own dense layers to the loaded model - we add identical ones to our model
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) #dodatkowa warstwa - połowa neuronów tej warstwy zostanie losowo wyłączona (w każdej iterazcji) / additional layer - half of the neurons in this layer will be randomly turned off (each iteration)
prediction_output = Dense(len(class_names), activation='softmax')(x)

#Nowy model
#New model
model2 = Model(inputs = base_model.inputs, outputs = prediction_output)

#Zamrożenie warstw bazowego modelu VGG16 - przeczytaj freeze.txt aby dowiedzieć się więcej dlaczego
##Freezing layers of the VGG16 base model - read freeze.txt for more details why
for layer in base_model.layers:
    layer.trainable = False

model2.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])#, run_eagerly=True) - optional see the description below
'''
[EN]
In "eager" mode, every operation is performed immediately. The main advantage of the "eager" mode is the ability
to debug the model on the fly, as you can see the results of each operation in real time, however, this
can lead to higher resource consumption and increased training time.

[PL]
W trybie "eager", każda operacja jest wykonywana natychmiastowo. Główną zaletą trybu "eager" jest możliwość
debugowania modelu na bieżąco, ponieważ możesz zobaczyć wyniki każdej operacji w czasie rzeczywistym, jednakże
może to prowadzić do większego zużycia zasobów i wydłużonego czasu treningu.
'''

train_images = np.load("TransformedData/train_images.npy")
train_labels = np.load("TransformedData/train_labels.npy")

history = model2.fit(train_images, train_labels, batch_size = 32, epochs = 10, validation_split = 0.2) #in last epoch: loss - 0.22, accuracy - 0.9176, val_loss - 0.3835, val_accuracy - 0.8785

test_loss, test_accuracy = model2.evaluate(test_images, test_labels) #loss - 0.3804, accuracy - 0.8737

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

predictions = model2.predict(test_images)
predicted_classes = np.argmax(predictions, axis = 1)

model_accuracy = accuracy_score(test_labels, predicted_classes)

print(f'Model accuracy: {model_accuracy}')



#Confiussion Matrix
cm = confusion_matrix(test_labels, predicted_classes)

cmap = sns.color_palette("inferno", as_cmap=True)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix\nAccuracy: {model_accuracy:.2f}")
plt.savefig("ModelTestingCharts/confusion_matrix.png")
plt.show()



plt.figure(figsize=(12, 6))

#Wykres funkcji straty / Loss function chart
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

#Wykres dokładności / accuracy chart
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("ModelTestingCharts/training_history.png")  # Zapisz jako plik PNG
plt.show()


#Zapisz model / Save model
filename = f"Models/TunedModel_VGG16-{round(model_accuracy, 2)}.h5"

model2.save(filename)





from CNN_Natural_Space_Images_Classification import display_random_image

display_random_image(class_names, test_images, predicted_classes)
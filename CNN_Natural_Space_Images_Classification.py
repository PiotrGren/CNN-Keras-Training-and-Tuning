import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.4)
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)


#WCZYTANIE DANYCH / READING DATA
def load_data():
    """
        Wczytujemy dane:
            - 14,034 zdjęć do trenowania sieci neuronowej.
            - 3,000 zdjęć do oceny jak dokładnie sieć nauczyła się klasyfikować obrazy.
    """
    datasets = ['Dataset/NSpaces/seg_train/seg_train', 'Dataset/NSpaces/seg_test/seg_test']
    output = []

    #Przechodzimy po koleii przez zbiór danych 
    #We go through the dataset one by one
    for dataset in datasets:
        images = []
        labels = []

        print("\nLoading {}".format(dataset))

        #Przechodzimy przez każdy folder odpowiadający kategorii
        #Go through each folder corresponding to the category
        for folder in os.listdir(dataset):
            label = class_names_label[folder] #Tutaj z naszego słownika zczytujemy liczbę będącą etykietą danej kategorii w zależności od nazwy kategorii (nazwy folderu)
                                              #Here we read from our dictionary the number that is the label of the category depending on the category name (folder name)

            #Przechodzimy po koleii po każdym zdjęciu w folderze
            #Go through each photo in the folder one by one
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                #Pobieramy ścieżkę do pliku
                #We download the path to the file
                img_path = os.path.join(os.path.join(dataset, folder), file)

                #Otwieramy obraz i przekształcamy jego rozmiar na taki jaki potrzebujemy (150x150)
                #Open the image and convert it to the size we need (150x150)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)
                image = image / 255.0

                #Dodajemy zmienione zdjęcie i odpowiadającą mu etykietę do outputu
                #Add the changed photo and corresponding label to the output
                images.append(image)
                labels.append(label)

        #Konwertujemy zdjęcia i etykiety na macierze numpy o odpowiednim typie danych
        #Convert images and labels to numpy arrays with the appropriate data type
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')

        #Przyłączenie przekonwerowanych zdjęć i etykiet do outputu funkcji
        #Appending converted photos and labels to function output
        output.append((images, labels))

    return output
    
(train_images, train_labels), (test_images, test_labels) = load_data()

#Przemieszanie danych do trenowania modelu w sposób losowy
#Shuffle training data
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

#Zapisanie danych do plików z odpowiednim rozszerzeniem npy
#Save data to files with the npy extension
np.save("TransformedData/train_images.npy", train_images)
np.save("TransformedData/train_labels.npy", train_labels)
np.save("TransformedData/test_images.npy", test_images)
np.save("TransformedData/test_labels.npy", test_labels)
np.save("TransformedData/class_names.npy", class_names)
np.save("TransformedData/class_labels.npy", class_names_label)

n_train = train_labels.shape[0]
n_test = test_labels.shape[0]
print("PL")
print("\n\nLiczba próbek do trenowania modelu: {}".format(n_train))
print("Liczba próbek do testowania modelu: {}".format(n_test))
print("Każde zdjęcie jest rozmiaru: {}".format(IMAGE_SIZE))

print("EN")
print("\n\nNumber of samples to train the model: {}".format(n_train))
print("Number of samples to test the model: {}".format(n_test))
print("Each photo is sized: {}".format(IMAGE_SIZE))




#WIZUALIZACJA ZAGADNIENIA
#VISUALIZATION OF THE ISSUE

#Wykres przedstawiający ilość zdjęć przeznaczonych do trenowania w porównaniu do tych przeznaczonych do testowania w podziale na kategorie
#Chart showing the number of images intended for training versus those intended for testing by category
_, train_counts = np.unique(train_labels, return_counts=True)
_, test_counts = np.unique(test_labels, return_counts=True)
pd.DataFrame({'train': train_counts, 'test': test_counts}, index=class_names).plot.bar()
plt.show()



#Wykres przedstawiający udział każdej kategorii w ogólnej liczbeności zbioru danych
#Chart showing the contribution of each category to the total count of the dataset
plt.pie(train_counts, explode=(0, 0, 0, 0, 0, 0), labels = class_names, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Udział zaobserwowanych kategorii w liczebności zbioru')
plt.show()


#Funkcja pokazująca 25 obrazków z macierzy obrazków oraz odpowiadające im etykiety
#Function to show 25 images from the image array and their corresponding labels
def display_examples(class_names, images, labels):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Przykłądy zdjęć ze zbioru danych", fontsize=16)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()

display_examples(class_names, train_images, train_labels)


#Funkcja pokazująca losowe zdjęcie ze zbioru wraz z jego numerem oraz etykietą
#Function to show a random photo from the collection along with its number and label
def display_random_image(class_names, images, labels):
    #Funkcja przedstawiająca llosowe zdjęcie z macierzy zdjęć oraz odpowiadającą mu etykietę z macierzy etykiet
    #Function showing llos photo from photo array and corresponding label from label array
    index = np.random.randint(images.shape[0])
    plt.figure()
    plt.imshow(images[index])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title("Image #{} : ".format(index) + class_names[labels[index]])
    plt.show()

display_random_image(class_names, train_images, train_labels)





#Budujemy / Inicjalizujemy Konwolucyjny Model Sieci Neuronowej [CNN]
#Building/Initializing a Convolutional Neural Network Model [CNN].
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = (150, 150, 3)), #Ustawiamy rozmiar wejściowy 150x150 tak jak przeskalowaliśmy zdjęcia / Set the input size to 150x150 as we scaled the photos
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax) #Warstwa zwracająca wektor prawdopodobieństwa (do której kategorii należy obrazek) / Layer that returns a probability vector (which category the image belongs to)
])

#Kompilacja modelu CNN z optymalizatorem Adam, funkcją straty jako kategoryczna entropia krzyżowa oraz dokładnością modelu jako metryką oceny modelu
#Compilation of CNN model with Adam optimizer, loss function as categorical cross entropy and model accuracy as model evaluation metric
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Trenujemy model w 20 epok po 128 próbek w pojedyńczej partii danych oraz z ustalonym zbiorem walidacyjnym dla każdej epoki, którym jest 20% zbioru treningowego
#We train the model in 20 epochs of 128 samples in a single batch of data and with a fixed validation set for each epoch, which is 20% of the training set
history = model.fit(train_images, train_labels, batch_size = 128, epochs = 20, validation_split = 0.2)

'''
#Wykres dokładności modelu
#Model accuracy chart
def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['acc'],'bo--', label = "acc")
    plt.plot(history.history['val_acc'], 'ro--', label = "val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()

plot_accuracy_loss(history)
'''


#Ocena modelu na podstawie danych testowych i przekazanie wyników do zmiennej
#Evaluate the model based on the test data and pass the results to the variable
test_loss = model.evaluate(test_images, test_labels)


predictions = model.predict(test_images)        #Tworzymy wektor prawdopodobieństw / Create a vector of probabilities 
pred_labels = np.argmax(predictions, axis = 1)  #Bierzemy największe prawdopodobieństwo z ostatniej warstwy / We take the highest probability from the last layer
accuracy = accuracy_score(test_labels, pred_labels)

print("Model Accuracy : {}".format(accuracy))

display_random_image(class_names, test_images, pred_labels)




#WYKRES ŹLE ZETYKIETOWANYCH ZDJĘĆ
#CHART OF MISLABELED PHOTOS
def print_mislabeled_images(class_names, test_images, test_labels, pred_labels):
    """
        Wykres pokazujący 25 losowych zdjęć ze źle podpisanych obrazów, tam gdzie test_labels != pred_labels
    """
    binary_labels = (test_labels == pred_labels)    #Jeżeli etykieta była dobrze nadana to będzie 1, a jak źle to 0
    mislabeled_indices = np.where(binary_labels == 0)
    mislabeled_images = test_images[mislabeled_indices]
    mislabeled_labels = pred_labels[mislabeled_indices]

    title = "Some examples of mislabeled images by the classifier:"
    display_examples(class_names,  mislabeled_images, mislabeled_labels)

print_mislabeled_images(class_names, test_images, test_labels, pred_labels)



#MACIERZ POMYŁEK
#CONFUSION MATRIX
CM = confusion_matrix(test_labels, pred_labels)
ax = plt.axes()
sns.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           xticklabels=class_names, 
           yticklabels=class_names, ax = ax)
ax.set_title('Confusion matrix')
plt.show()



#Zapisanie Wytrenowanego Modelu
#Save The Trained Model
subject = 'TrainedModel_NS'
save_id = f"{subject}-{accuracy:5.2f}.h5"
model_save_loc = os.path.join('Models', save_id)
model.save(model_save_loc)
print("Model was saved as {}".format(model_save_loc))


import tensorflow as tf
from tensorflow.keras import datasets,layers,models
from matplotlib import pyplot as plt
import numpy as np
show_images = False

## load and normalize the dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
train_images, test_images = train_images /255.0, test_images /255.0

## define the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

if (show_images):
    plt.figure(figsize=(10,10))
    for i in range(20):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()
    

### define the model
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
## train and evaluate the model
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(f"Test dataset accuracy: {test_acc} and loss: {test_loss}")


#### make predictions
predictions_array = model.predict(test_images)
top_predictions = np.argmax(predictions_array, axis=1)
print("Predicted Classes: ",[class_names[i] for i in top_predictions[:10].tolist()])
print("True Classes: ",[class_names[i] for i in test_labels[:10].tolist()])

### show the probabilties of these predictions
model_predictions = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions_probabilities = model_predictions.predict(test_images)
predictions_probabilities = np.max(predictions_probabilities, axis=1)
print("Prbababilities of these predictions: ",predictions_probabilities[:10])
#denseNet
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import time

# Load data
mfcc_train = np.load('mel_train.npy')
mfcc_test = np.load('mel_test.npy')
labels_train = np.load('labels_train.npy')
labels_test = np.load('labels_test.npy')

# Define DenseNet model
# def dense_block(x, blocks, name):
#     for i in range(blocks):
#         x1 = conv_block(x, 32, name=name + '_block' + str(i + 1))
#         x = layers.Concatenate(axis=-1, name=name + '_concat')([x, x1])  # Concatenate along the channel axis
#     return x


# def transition_block(x, reduction, name):
#     bn_axis = 3
#     x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
#     x = layers.Activation('relu', name=name + '_relu')(x)
#     x = layers.Conv2D(int(tf.keras.backend.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
#                       name=name + '_conv')(x)
#     x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
#     return x

# def conv_block(x, growth_rate, name):
#     bn_axis = 3
#     x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
#     x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
#     x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
#     x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
#     x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
#     x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
#     x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
#     return x

# def FiveLayerDenseNet(input_shape=(150, 365, 1), classes=13):
#     growth_rate = 32
#     compression_factor = 0.5

#     # Define input layer
#     inputs = layers.Input(shape=input_shape)

#     # Initial convolutional layer
#     x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)

#     # Dense blocks with 2 convolutional layers each
#     for _ in range(4):
#         x1 = layers.Conv2D(growth_rate, kernel_size=(3, 3), activation='relu', padding='same')(x)
#         x1 = layers.Conv2D(growth_rate, kernel_size=(3, 3), activation='relu', padding='same')(x1)
#         x = layers.Concatenate()([x, x1])
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Conv2D(int(x.shape[-1] * compression_factor), kernel_size=(1, 1), activation='relu')(x)
#         x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     # Global average pooling and output layer
#     x = layers.GlobalAveragePooling2D()(x)
#     # Additional neural network layers after global average pooling
#     x = layers.Dense(128, activation='relu')(x)
#     x = layers.Dropout(0.25)(x)  # Adding dropout for regularization
#     x = layers.Dense(128, activation='relu')(x)

#     outputs = layers.Dense(classes, activation='softmax')(x)

#     # Create model
#     model = models.Model(inputs, outputs, name='FiveLayerDenseNet')
#     return model



# # Create DenseNet model
# model = FiveLayerDenseNet()


labels_train = np.array(labels_train)
labels_test = np.array(labels_test)

# Define model architecture
model = models.Sequential([
    layers.Input(shape=(20, 365, 1)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(13, activation='softmax')
])


# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Reshape data for CNN input
# mfcc_train = mfcc_train.reshape(-1, 20, 365, 1)
# mfcc_test = mfcc_test.reshape(-1, 20, 365, 1)


for epoch in range(1, 151):
    try:
        print(f'epoch: {epoch}, batch size: 128')
        model.fit(mfcc_train, labels_train, epochs=1, batch_size=128, verbose=0)
        train_loss, train_accuracy = model.evaluate(mfcc_train, labels_train, verbose=0)
        print(f'train Loss: {train_loss}')
        print(f'train Accuracy: {train_accuracy}')
        test_loss, test_accuracy = model.evaluate(mfcc_test, labels_test, verbose=0)
        print(f'Test Loss: {test_loss}')
        print(f'Test Accuracy: {test_accuracy}')
        if(test_accuracy>0.88):
            path = "mfcc_128_"+str(epoch)+".keras"
            model.save(path)
    except Exception as e:
        print(f"Training failed: {str(e)}")


# Make predictions on the test data
predictions = model.predict(mfcc_test)

# Convert probability predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_labels == labels_test)

# Calculate confusion matrix
confusion_matrix = tf.math.confusion_matrix(labels_test, predicted_labels)

# Calculate true positives, false positives, and false negatives
true_positives = np.diag(confusion_matrix)
false_positives = np.sum(confusion_matrix, axis=0) - true_positives
false_negatives = np.sum(confusion_matrix, axis=1) - true_positives

# Calculate precision, recall, and F1-score
precision = np.mean(true_positives / (true_positives + false_positives))
recall = np.mean(true_positives / (true_positives + false_negatives))

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Calculate true positives, false positives, and false negatives for each class
class_true_positives = np.diag(confusion_matrix)
class_false_positives = np.sum(confusion_matrix, axis=1) - class_true_positives
class_false_negatives = np.sum(confusion_matrix, axis=0) - class_true_positives

# Calculate precision, recall, and accuracy for each class
class_precision = class_true_positives / (class_true_positives + class_false_positives)
class_recall = class_true_positives / (class_true_positives + class_false_negatives)
class_accuracy = class_true_positives / np.sum(confusion_matrix, axis=1)

# Print class-wise metrics
for i in range(num_classes):
    print("Class:", i)
    print("Precision:", class_precision[i])
    print("Recall:", class_recall[i])
    print("Accuracy:", class_accuracy[i])
    print()
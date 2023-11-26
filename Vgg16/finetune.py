# %%
import tensorflow as tf
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# %%
def load_data(base_dir):
    image_paths = []
    labels = []
    label_map = {}
    current_label = 0

    for i in range(1, 48):
        if i == 4:
            continue

        for side in ['left', 'right']:
            side_dir = os.path.join(base_dir, str(i), side)
            if os.path.exists(side_dir):
                images = os.listdir(side_dir)
                for image in images:
                    if image.endswith('.bmp'):
                        image_path = os.path.join(side_dir, image)
                        label = str(i) + '_' + side
                        if label not in label_map:
                            label_map[label] = current_label
                            current_label += 1
                        image_paths.append(image_path)
                        labels.append(label_map[label])

    return image_paths, labels


# %%
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_bmp(image, channels=3)

    image = tf.image.resize(image, [224, 224]) # VGG16 224x224

    image = tf.keras.applications.vgg16.preprocess_input(image)

    return image, label

# %%
image_paths, labels = load_data("MMU-Iris-Database/")

image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
label_dataset = tf.data.Dataset.from_tensor_slices(labels)
dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

dataset = dataset.map(preprocess_image)
dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(32)


# %%
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(320, 240, 3))

x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# %%
model.fit(dataset, epochs=10)

# %%
model.save('VGG-16')



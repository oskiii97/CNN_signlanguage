import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from configparser import ConfigParser
import random

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import layers
from keras import Sequential
from sklearn.metrics import classification_report


# function for copy data to the new folder and split for train, valid and testing.
def split_the_data(data_path, cls_labels, train_ratio, valid_ratio):
    try:
        # creating folder for images and copy data folder content
        if os.path.exists("imgs"):
            os.makedirs("imgs")
        shutil.copytree(data_path, "imgs")
        data_path = data_path.replace("data", "imgs")

        split_folders = ['train', 'valid', 'test']
        # creating folders test/valid/train
        for cls in cls_labels:
            os.makedirs(os.path.join(data_path, split_folders[0], cls), exist_ok=True)
            os.makedirs(os.path.join(data_path, split_folders[1], cls), exist_ok=True)
            os.makedirs(os.path.join(data_path, split_folders[2], cls), exist_ok=True)

        for cls in cls_labels:
            images = os.listdir(os.path.join(data_path, cls))
            random.shuffle(images)  # shuffling of files
            num_train = int(len(images) * train_ratio)
            num_val = int(len(images) * valid_ratio)
            train_files = images[:num_train]
            val_files = images[num_train:num_train + num_val]
            test_files = images[num_train + num_val:]
            for label in cls_labels:
                for img in train_files:
                    src = os.path.join(data_path, label, img)
                    dst = os.path.join(data_path, str(split_folders[0]), str(label), str(img))
                    shutil.copyfile(src, dst)
                for img in val_files:
                    src = os.path.join(data_path, label, img)
                    dst = os.path.join(data_path, str(split_folders[1]), str(label), str(img))
                    shutil.copyfile(src, dst)
                for img in test_files:
                    src = os.path.join(data_path, label, img)
                    dst = os.path.join(data_path, str(split_folders[2]), str(label), str(img))
                    shutil.copyfile(src, dst)

            num_all_train = num_train * len(cls_labels)
            num_all_valid = num_val * len(cls_labels)
            # delete cls folders
            for cls_folder in cls_labels:
                shutil.rmtree(os.path.join(data_path, cls_folder))

            train_path = os.path.join(data_path, split_folders[0])
            valid_path = os.path.join(data_path, split_folders[1])
            test_path = os.path.join(data_path, split_folders[2])

            return train_path, valid_path, test_path, num_all_train, num_all_valid
    except Exception as error:
        print(f"Error during function split_the_data, error: {error}")


# function to show metrics and analyzing testing and valid metrics
def metrics_plot(history_df):  # hist is data frame with training results
    try:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(history_df['epoch'], history_df['accuracy'], label='Accuracy')
        plt.plot(history_df['epoch'], history_df['loss'], label='Loss')
        plt.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy')
        plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')

        plt.title('Accuracy vs Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Values')
        plt.legend()
        plt.show()
    except Exception as error:
        print(f"Error during function split_the_data, error: {error}")


# Creating ConfigParser object and getting data path from config
config_object = ConfigParser()
config_object.read("config.ini")
data_path = config_object["PATHS"]["datapath"]

# getting class labels
cls_labels = os.listdir(data_path)

# Check how many imgs are in the folder
number_of_files = {}
for folder in cls_labels:
    number_of_files[folder] = len(os.listdir(os.path.join(data_path, folder)))

print(f'Count of the files is: {number_of_files}')

# ratio of data types
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# check if files are previously split
if os.path.exists("imgs"):
    shutil.rmtree("imgs")
    train_path, valid_path, test_path, num_train, num_valid = split_the_data(data_path,
                                                                             cls_labels, train_ratio, valid_ratio)
else:
    train_path, valid_path, test_path, num_train, num_valid = split_the_data(data_path,
                                                                             cls_labels, train_ratio, valid_ratio)

# data augmentation and normalization
train_data_generator = ImageDataGenerator(
    rescale=1./255.,  # data normalization
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')
# valid data should be only rescaled
valid_data_generator = ImageDataGenerator(rescale=1./255.)

# prepare images to putting inside model
train_generator = train_data_generator.flow_from_directory(directory=train_path,
                                                           target_size=(100, 100),
                                                           batch_size=2,
                                                           class_mode="categorical")
# categorical because there is more than 2 classes

valid_generator = train_data_generator.flow_from_directory(directory=valid_path,
                                                           target_size=(100, 100),
                                                           batch_size=2,
                                                           class_mode="categorical")

# building CNN model
batch_size = 2
steps_per_epoch = num_train // batch_size
valid_steps = num_valid // batch_size

model = Sequential()
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D(pool_size=(4, 4)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(units=256, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(units=32, activation='relu'))
model.add(layers.Dense(units=4, activation='softmax'))  # softmax because more than 2 classes

# compiling of the model
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics='accuracy')
# training model
history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=15,
                    validation_data=valid_generator,
                    validation_steps=valid_steps,
                    )

# save model to file
model.save('model.h5')

# creating dataframe with results and showing metrics on splot
history_df = pd.DataFrame(history.history)
history_df['epoch'] = history.epoch
metrics_plot(history_df)

# evaluation of the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_directory(test_path,
                                                  target_size=(100, 100),
                                                  batch_size=1,
                                                  class_mode='categorical',
                                                  shuffle=False
                                                  )
y_prob = model.predict_generator(test_generator, test_generator.samples)
# predicts
y_pred = np.argmax(y_prob, axis=1)
predictions_df = pd.DataFrame({'class': y_pred})

y_true = test_generator.classes  # real classes from generator
y_pred = predictions_df['class'].values

# get 10 random values
indexes = np.random.choice(len(y_pred), size=10, replace=False)

# create plot with 10 imgs
random_idx = np.random.choice(len(y_pred), 10, replace=False)

fig, ax = plt.subplots(figsize=(8, 6))

for i in random_idx:
    color = 'green' if y_pred[i] == y_true[i] else 'red'
    ax.scatter(i, y_pred[i], color=color)

ax.set_xticks(random_idx)
ax.set_xticklabels(random_idx)
ax.set_xlabel('INDEX')
ax.set_ylabel('CLASS')
ax.set_title('10 RANDOM PREDICTIONS')

ax.scatter([], [], color='green', label='GOOD PREDICTION')
ax.scatter([], [], color='red', label='BAD PREDICTION')
ax.legend()
ax.set_yticks(range(len(cls_labels)))
ax.set_yticklabels(cls_labels)

# print classification report
print(classification_report(y_true, y_pred, target_names=cls_labels))

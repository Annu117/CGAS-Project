import time 
import pandas as pd
# import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout,Flatten,Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from keras.regularizers import l2


# Disable GPU memory growth
tf.config.set_visible_devices([], 'GPU')

data = pd.read_csv('data.csv')

# Map class names to indices
class_names = data['Label'].unique()
print(data['Label'])   # Printing the labels 


# Split data into training and testing
train_data, test_data = train_test_split(data, test_size=0.25, random_state=53, stratify=data['Label'])
print("========================= Train data =================================")
print(train_data)
print("========================= Test Data ========================================= ")
print(test_data)

def learning_rate_schedule(epoch):
    if epoch < 5:
        return 1e-3
    elif 5 <= epoch < 10:
        return 2e-4
    elif 10 <= epoch < 15:
        return 2e-5
    else:
        return 5e-7

learning_scheduler = LearningRateScheduler(learning_rate_schedule)

# Image Data Generators
train_data_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,fill_mode='nearest', rotation_range=40, width_shift_range=0.2 )
test_data_gen = ImageDataGenerator(rescale=1. / 255)


# Function to load data from CSV for ImageDataGenerator
def create_data_gen(dataframe, datagen, target_size, batch_size):
    return datagen.flow_from_dataframe(
        dataframe,
        x_col="Filepath",
        y_col="Label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )


# Create generators
batch_size = 32
target_size = (224, 224)
print("\n========================= Train and Test Image Generator ================================")
train_gen = create_data_gen(train_data, train_data_gen, target_size, batch_size)
test_gen = create_data_gen(test_data, test_data_gen, target_size, batch_size)


# =========================== Model Architecture : InceptionV3 =========================


inception = InceptionV3(weights='imagenet', include_top=False,input_tensor=Input(shape=(224, 224, 3)))
layer = inception.output
layer = AveragePooling2D(pool_size=(2, 2))(layer)
layer = Dropout(0.5)(layer)
layer = Flatten()(layer)
# layer = Dense(128, activation='relu')(layer)

predictions = Dense(len(class_names), kernel_initializer='glorot_uniform',kernel_regularizer=l2(.0005), activation='softmax')(layer)
model = Model(inputs=inception.input, outputs=predictions)

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


# Callbacks
checkpointer = ModelCheckpoint(filepath='best_model.h5', save_best_only=True)
csv_logger = CSVLogger('history.log')

# model.summary()

# Train the model
print("\n Training Started...... ")
t1 = time.time()


history = model.fit(
    train_gen,
    steps_per_epoch=len(train_data) // batch_size,
    validation_data=test_gen,
    validation_steps=len(test_data) // batch_size,
    epochs=30,
    callbacks=[learning_scheduler,csv_logger, checkpointer]
)

# Save the final model
model.save('final_model.h5')
t2 = time.time()
T = ((t2-t1)/60)
print("Total Time taken for Training : ",T/60, "Hours")


# Step 1: Load the log file into a DataFrame
log_file = "history.log"  # Replace with the actual file path
history = pd.read_csv(log_file)

# Step 2: Plot Training and Validation Accuracy
plt.style.use('ggplot')  # Replace with a preferred style
plt.figure(figsize=(10, 5))
plt.plot(history['epoch'], history['accuracy'], marker='o', linestyle='dashed', label='Train Accuracy')
plt.plot(history['epoch'], history['val_accuracy'], marker='x', linestyle='dashed', label='Validation Accuracy')
plt.title('FOOD_101 Dataset - Inceptionv3 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.yticks(np.arange(0, 1.1, 0.1))  # Set y-axis points with an interval of 0.1
plt.grid()
plt.show()

# Step 3: Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history['epoch'], history['loss'], marker='o', linestyle='dashed', label='Train Loss')
plt.plot(history['epoch'], history['val_loss'], marker='x', linestyle='dashed', label='Validation Loss')
plt.title('FOOD_101 Dataset - Inceptionv3 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')

plt.grid()
plt.show()



model = load_model('best_model.h5')
images_df = pd.read_csv('data.csv')
Image_Classes = list(images_df['Label'].unique())
Image_Classes.sort()


def get_real_and_predicted(test_data, model,  img_height=224, img_width=224):
    real = []  
    predicted = []  

 
    for index, row in test_data.iterrows():
        
        true_label = row['Label']
        
        # Load and preprocess the image
        img_path = row['Filepath']
        img_ = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img_)
        img_processed = np.expand_dims(img_array, axis=0)
        img_processed /= 255.  # Normalize the image pixels

        # Make a prediction using the model
        prediction = model.predict(img_processed)
        index_pred = np.argmax(prediction)  # Get the predicted class index
        pred_value = Image_Classes[index_pred]

        # Append the true and predicted labels to their respective lists
        real.append(true_label)
        predicted.append(pred_value)

    return real, predicted

def print_confusion_matrix(real,predicted):

    cmap="viridis"
    cm_plot_labels = [i for i in Image_Classes ]

    cm = confusion_matrix(y_true=real, y_pred=predicted)
    df_cm = pd.DataFrame(cm,cm_plot_labels,cm_plot_labels)
    
    plt.figure(figsize = (22,15))
    sns.set(font_scale=0.7)

    # Create the heatmap
    s = sns.heatmap(df_cm, annot=False, cmap=cmap)

   
    s.set_xticks(range(len(cm_plot_labels)))  # Set the positions for class names on x-axis
    s.set_yticks(range(len(cm_plot_labels)))  # Set the positions for class names on y-axis
    s.set_xticklabels(cm_plot_labels, rotation=70)  # Set the actual class labels on x-axis
    s.set_yticklabels(cm_plot_labels, rotation=0)  # Set the actual class labels on y-axis

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Food_101_confusion_matrix.png')
    plt.show()

t1 = time.time()

y_true,y_pred=get_real_and_predicted(test_data, model,  img_height=224, img_width=224)
print_confusion_matrix(y_true,y_pred)

t2 = time.time()

t = (t2-t1)/60

print("Time taken : ",t ,"Minutes")








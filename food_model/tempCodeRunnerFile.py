tf.config.set_visible_devices([], 'GPU')

# # print("TensorFlow version:", tf.__version__)
# # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# # Load data from CSV
# data = pd.read_csv('data.csv')

# # Map class names to indices
# class_names = data['Label'].unique()
# print(data['Label'])   # Printing the labels 


# # Split data into training and testing
# train_data, test_data = train_test_split(data, test_size=0.25, random_state=53, stratify=data['Label'])
# print("========================= Train data =================================")
# print(train_data)
# print("========================= Test Data ========================================= ")
# print(test_data)

# def learning_rate_schedule(epoch):
#     if epoch < 5:
#         return 1e-3
#     elif 5 <= epoch < 10:
#         return 2e-4
#     elif 10 <= epoch < 15:
#         return 2e-5
#     else:
#         return 5e-7

# learning_scheduler = LearningRateScheduler(learning_rate_schedule)

# # Image Data Generators
# train_data_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,fill_mode='nearest', rotation_range=40, width_shift_range=0.2 )
# test_data_gen = ImageDataGenerator(rescale=1. / 255)


# # Function to load data from CSV for ImageDataGenerator
# def create_data_gen(dataframe, datagen, target_size, batch_size):
#     return datagen.flow_from_dataframe(
#         dataframe,
#         x_col="Filepath",
#         y_col="Label",
#         target_size=target_size,
#         batch_size=batch_size,
#         class_mode="categorical"
#     )


# # Create generators
# batch_size = 32
# target_size = (224, 224)
# print("\n========================= Train and Test Image Generator ================================")
# train_gen = create_data_gen(train_data, train_data_gen, target_size, batch_size)
# test_gen = create_data_gen(test_data, test_data_gen, target_size, batch_size)


# required libraries
import numpy as np
import pandas as pd
from pathlib import Path
import os.path

"""
WE WILL CREATE A CSV FILE VIA TRAVERSING EACH IMAGE CATEGORY DIRECTORY. 
CSV FILE WILL HAVE TWO COLUMNS ONE FOR PATH OF THE IMAGE AND OTHER FOR THE LABEL OF THE IMAGE.
"""

# Food_101 directory path
image_dir = Path(r"C:\Users\Dell\OneDrive\Desktop\Indian Food\Indian Food Images")

#  File data Frames
filepaths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

images_df = pd.concat([filepaths, labels], axis=1)

print(images_df)
print("\n=================================================================================")
print("\nClasses(Label) and no of images to each class:\n")
print(images_df['Label'].value_counts())


# Saving the DataFrame to CSV file data.csv
images_df.to_csv("indian_data.csv", index=False)
print("\nData file food_data.csv saved successfully\n")


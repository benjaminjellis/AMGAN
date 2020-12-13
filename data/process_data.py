from os import listdir, mkdir
from os.path import isfile, exists
from pathlib import Path
from shutil import copyfile

path = str(Path().absolute())
raw_data_dir = path + "/raw/"

raw_images = [raw_data_dir + f for f in listdir(raw_data_dir) if isfile(raw_data_dir + f)]

output_dir = path + "/processed/"

for i in range(len(raw_images)):
    if not exists(output_dir):
        mkdir(output_dir)
    copyfile(raw_images[i], output_dir + "image_" + str(i + 1) + ".jpg")

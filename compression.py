# %%
from PIL import Image, ImageFile
import os
from tqdm import tqdm
# the following is for https://github.com/python-pillow/Pillow/issues/3185
ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%
# each image is resized (same ratio) and compressed into a new folder
dir_org = 'data/original'  # folder with the original images
dir_compressed = 'data/compressed'  # folder for the compressed images
fixed_height = 1024  # fixed height after resizing
quality = 65  # percentage of the compression, e.g., 65%

# %%
# find all jpg files
files = os.listdir(dir_org)  # all files in the folder
images = [file.lower() for file in files if file.endswith(('JPG'))]  # only jpg files
images

# %%
# resize and compress each image
count = 0
for image in tqdm(images):

    compressed_image = dir_compressed + '/' + image  # the path for the compressed image
    if not os.path.exists(compressed_image):  # skip processed images
        #print(f'processing {image}')

        try:
            # open the image
            im = Image.open(dir_org + '/' + image)

            # convert to RGB
            # RGBA cannot be saved into JPG https://stackoverflow.com/questions/48248405/cannot-write-mode-rgba-as-jpeg
            im = im.convert('RGB')  
            
            # resize with the same ratio
            height_percent = (fixed_height / float(im.size[1]))
            width_size = int((float(im.size[0]) * float(height_percent)))
            im = im.resize((width_size, fixed_height), Image.NEAREST)

            # compress
            im.save(dir_compressed + '/' + image, optimize=True, quality=quality)

            count += 1
        except Exception as ex:
            print(f'something wrong with {image}', ex)
            break  # stop

print(f'complete processing {count} images')



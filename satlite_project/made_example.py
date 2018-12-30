from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, merge, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

#from data import load_train_data, load_test_data
from osgeo import gdal


K.set_image_data_format('channels_last')  # TF dimension ordering in this code



smooth = 1.




def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols,11))
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(pool1)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu")(pool2)
    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu")(pool3)
    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu")(pool4)
    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu")(conv5)


    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=2)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=2)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=2)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=2)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    #imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])



    shapefiles = [os.path.join(validation_data_path, "%s.shp" % c)
                  for c in classes]
    verification_pixels = vectors_to_raster(shapefiles, rows, cols,
                                            geo_transform, proj)
    for_verification = np.nonzero(verification_pixels)
    verification_labels = verification_pixels[for_verification]


    imgs_test = verification_labels


    print('-'*30)
    print('Loading and preprocessing happy_song data...')
    print('-'*30)
    #imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)
    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on happy_song data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)


    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'

    classification = imgs_mask_test.reshape((rows,cols))
    predicted_labels = classification[for_verification]
    imgs_id_test = predicted_lables

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)



# A list of "random" colors (for a nicer output)
COLORS = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941"]


'''---------------------------------Set the Path and Variable-----------------------------'''

os.chdir("C:/Users/LSH/PycharmProjects/ML/LSH_Project")

output_fname = "Result/4/classification.tiff"
train_data_path = "Real_Data/happy_song/2/"
validation_data_path = "Real_Data/train/2/"
raster_data_path = "Real_Data/image/3/clip.tif"

raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
geo_transform = raster_dataset.GetGeoTransform()
proj = raster_dataset.GetProjectionRef()

bands_data = []

'''---------------------------------Set the Path and Variable-----------------------------'''



'''---------------------------------Data Processing-----------------------------'''


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """Rasterize the vectors in the given directory in a single image."""
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i + 1
        ds = create_mask_from_vector(path, cols, rows, geo_transform,
                                     projection, target_value=label)
        band = ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
        ds = None
    return labeled_pixels


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
                            projection, target_value=1):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')  # In memory dataset
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


def write_geotiff(fname, data, geo_transform, projection):
    """Create a GeoTIFF file with the given data."""
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    dataset = None  # Close the file


'''---------------------------------Data Processing-----------------------------'''

'''---------------------------------Training-----------------------------'''

'''
for b in range(1, raster_dataset.RasterCount+1):
    band = raster_dataset.GetRasterBand(b)
    bands_data.append(band.ReadAsArray())

    raster_data_path = "Real_Data/image/2/without9.tif"

raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
'''

import time
import skimage.io

for b in range(1, raster_dataset.RasterCount + 1):
    t1 = time.time()
    band = raster_dataset.GetRasterBand(b)
    bands_data.append(band.ReadAsArray())
    t2 = time.time()
    print(t2 - t1)
'''

    train_data_path = os.path.join(data_path, 'train')

    images = os.listdir(train_data_path)

    total = len(images) / 2



    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)


'''

bands_data = np.dstack(bands_data)
rows, cols, n_bands = bands_data.shape
img_rows, img_cols = rows, cols


files = [f for f in os.listdir(train_data_path) if f.endswith('.shp')]
classes = [f.split('.')[0] for f in files]
shapefiles = [os.path.join(train_data_path, f)
              for f in files if f.endswith('.shp')]

raster = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
labeled_pixels = raster
is_train = np.nonzero(labeled_pixels)
training_labels = labeled_pixels[is_train]
training_samples = bands_data[is_train]

imgs_train = training_samples
imgs_mask_train = training_labels

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)
#imgs_train, imgs_mask_train = load_train_data()
'''
imgs_train = preprocess(imgs_train)
imgs_mask_train = preprocess(imgs_mask_train)
'''
imgs_train = imgs_train.astype('float32')
mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean
imgs_train /= std

imgs_mask_train = imgs_mask_train.astype('float32')
imgs_mask_train /= 255.  # scale masks to [0, 1]
print('-'*30)
print('Creating and compiling model...')
print('-'*30)

model = get_unet()
model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

print('-'*30)
print('Fitting model...')
print('-'*30)
model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
          validation_split=0.2,
          callbacks=[model_checkpoint])



shapefiles = [os.path.join(validation_data_path, "%s.shp" % c)
              for c in classes]
verification_pixels = vectors_to_raster(shapefiles, rows, cols,
                                        geo_transform, proj)
for_verification = np.nonzero(verification_pixels)
verification_labels = verification_pixels[for_verification]


imgs_test = verification_labels


print('-'*30)
print('Loading and preprocessing happy_song data...')
print('-'*30)
#imgs_test, imgs_id_test = load_test_data()
imgs_test = preprocess(imgs_test)
imgs_test = imgs_test.astype('float32')
imgs_test -= mean
imgs_test /= std

print('-'*30)
print('Loading saved weights...')
print('-'*30)
model.load_weights('weights.h5')

print('-'*30)
print('Predicting masks on happy_song data...')
print('-'*30)
imgs_mask_test = model.predict(imgs_test, verbose=1)


np.save('imgs_mask_test.npy', imgs_mask_test)

print('-' * 30)
print('Saving predicted masks to files...')
print('-' * 30)
pred_dir = 'preds'

classification = imgs_mask_test.reshape((rows,cols))
predicted_labels = classification[for_verification]
imgs_id_test = predicted_lables

if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
for image, image_id in zip(imgs_mask_test, imgs_id_test):
    image = (image[:, :, 0] * 255.).astype(np.uint8)
    imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import gdown
import os
import h5py
import tqdm
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
import tensorflow as tf
import keras.api.backend as K
from keras.api.applications.vgg16 import VGG16
from keras.api.applications import ResNet50
from skimage.restoration import denoise_bilateral
from keras.api.callbacks import ReduceLROnPlateau, Callback, EarlyStopping
from skimage import exposure
from skimage.filters import median
from tensorflow.python import keras
from keras.api.layers import Layer
from keras.api.models import Model
from keras.api.optimizers import Adamax
from PIL import Image
import io
from fastapi.responses import JSONResponse
import base64

fx = 3
VGG=VGG16(input_shape=(256*fx,256*fx,3),weights='imagenet', include_top=False)
VGG.trainable = False

ResNet50_= ResNet50(input_shape=(256*fx,256*fx,3),weights='imagenet', include_top=False)
ResNet50_.trainable = False
upsampler = layers.UpSampling2D(size=(fx, fx), interpolation='lanczos5')


BinaryCrossentropy_loss = keras.losses.BinaryCrossentropy(label_smoothing=0.1)

def psnr(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mse = tf.where(tf.equal(mse, 0), 1e-10, mse)  # Avoid zero division
    max_pixel = 255.0
    psnr = 20 * tf.math.log(max_pixel / tf.sqrt(mse)) / tf.math.log(10.0)
    return psnr


def VGG_loss(y_true, y_pred):
      y_true = VGG(tf.repeat(upsampler(y_true), 3, axis=-1))
      y_pred = VGG(tf.repeat(upsampler(y_pred), 3, axis=-1))

      h1 = k.batch_flatten(y_true)
      h2 = K.batch_flatten(y_pred)
      rc_loss =  K.sum(K.square(h1 - h2), axis=-1)
      return rc_loss

def ResNet50_loss(y_true, y_pred):

      y_true = ResNet50_(tf.repeat(upsampler(y_true), 3, axis=-1))
      y_pred = ResNet50_(tf.repeat(upsampler(y_pred), 3, axis=-1))

      h1 = K.batch_flatten(y_true)
      h2 = K.batch_flatten(y_pred)
      rc_loss =  K.sum(K.square(h1 - h2), axis=-1)
      return rc_loss

def BC_loss(y_true, y_pred):
      y_true = upsampler(y_true)
      y_pred = upsampler(y_pred)
      BC_loss = BinaryCrossentropy_loss(y_true, y_pred)
      return BC_loss


def ssim_metric(y_true, y_pred):
     y_true = (upsampler(y_true))
     y_pred = (upsampler(y_pred))
     return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))



#### data for input included, T1, Flair, DWI .. all in 3ch array
# test_data_path = 'Patients/'
# input_ar = np.load("{}input.npy".format(test_data_path))
# T1 = np.load("{}T1.npy".format(test_data_path))
# Flair = np.load("{}Flair.npy".format(test_data_path))
# T2 = np.load("{}T2.npy".format(test_data_path))
# Contrast = np.load("{}Contrast.npy".format(test_data_path))


# output_ar = Contrast


############ Generator
## Define the generator network

def channel_attention(input_, ratio = 8):
    batch_, w, h, channel = input_.shape

    ## Global average pooling
    x1 = layers.GlobalAveragePooling2D()(input_)
    x1 = layers.Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x1)
    x1 = layers.Dense(channel, use_bias=False, kernel_initializer='he_normal')(x1)

    ## Max pooling
    x2 = layers.GlobalMaxPool2D()(input_)
    x2 = layers.Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x2)
    x2 = layers.Dense(channel, use_bias=False, kernel_initializer='he_normal',)(x2)

    ### Add both and apply sigmoid
    features = x1 + x2
    features = layers.Activation('sigmoid')(features)
    features = layers.multiply([input_, features])

    return features


class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        # Define the Conv2D layer in the constructor
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, input_):
        # Apply spatial attention logic
        x1 = tf.reduce_mean(input_, axis=-1, keepdims=True)  # Average pooling
        x2 = tf.reduce_max(input_, axis=-1, keepdims=True)  # Max pooling
        concat = tf.concat([x1, x2], axis=-1)
        x = self.conv(concat)  # Use the pre-defined Conv2D layer
        return input_ * x  # Element-wise multiplication

def CBAM(input_): #Convolutional Block Attention Module (CBAM)
    X = channel_attention(input_)
    X = SpatialAttention()(X)
    return X

def encoder_block(input_, f):
    CN = layers.Conv2D(f, (7,7), activation='leaky_relu', padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(input_)
    CN = layers.Conv2D(f, (3,3), activation='leaky_relu', padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(CN)

    ### dilatation Conv
    CD = layers.Conv2D(f*2, (3,3), dilation_rate=3, activation='leaky_relu', padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(CN)
    CD = layers.Conv2D(f*2, (3,3), dilation_rate=3, activation='leaky_relu',padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(CD)
    CD = layers.Conv2D(f, (3,3), dilation_rate=3, activation='leaky_relu',padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(CD)

    C = layers.Conv2D(f, (3,3), activation='leaky_relu', padding='same',kernel_initializer='he_normal', kernel_regularizer='l1')(CN)
    C = layers.concatenate([C, CN])
    C = layers.BatchNormalization()(C)

    skip_connect = layers.Dropout(0.05)(C)
    forward = layers.MaxPooling2D((2,2))(C)
    return forward, skip_connect

def decoder_block(input_, skip_, f):
    U = layers.Conv2DTranspose(f, (2,2), strides=(2,2), activation='leaky_relu', padding='same',kernel_initializer='he_normal', kernel_regularizer='l1')(input_)
    C = layers.BatchNormalization()(U)
    C = layers.Dropout(0.5)(C)
    C = layers.concatenate([C, skip_])
    C = layers.Conv2D(f, (3,3), dilation_rate=3, activation='leaky_relu', padding='same',kernel_initializer='he_normal', kernel_regularizer='l1')(C)
    C = layers.Conv2D(f, (3,3), dilation_rate=3, activation='leaky_relu', padding='same', kernel_initializer='he_normal', kernel_regularizer='l1', )(C)
    forward = layers.BatchNormalization()(C)
    return forward


def build_generator(images_shape, f, encoder_trainable = True):
    inp = layers.Input(images_shape)

    encoder_1 = encoder_block(inp, f)
    encoder_2 = encoder_block(encoder_1[0], f)
    encoder_3 = encoder_block(encoder_2[0], f*2)
    encoder_4 = encoder_block(encoder_3[0], f*4)
    encoder_5 = encoder_block(encoder_4[0], f*8)
    encoder_6 = encoder_block(encoder_5[0], f*16)


    neck_in = layers.Conv2D(f*32, (3,3), activation='relu', padding='same', name='Model_Neck_1',kernel_initializer='he_normal', kernel_regularizer='l1')(encoder_6[0])
    neck = layers.Conv2D(f*32, (3,3), activation='relu', padding='same', name='Model_Neck_2',kernel_initializer='he_normal', kernel_regularizer='l1')(neck_in)
    neck = layers.Conv2D(f*32, (3,3), activation='relu', padding='same', name='Model_Neck_3',kernel_initializer='he_normal', kernel_regularizer='l1')(neck)
    neck_out = CBAM(neck)

    decoder_6 = decoder_block(neck_out, CBAM(encoder_6[1]), f*16)
    decoder_5 = decoder_block(decoder_6, CBAM(encoder_5[1]), f*8)
    decoder_4 = decoder_block(decoder_5, CBAM(encoder_4[1]), f*4)
    decoder_3 = decoder_block(decoder_4, CBAM(encoder_3[1]), f*2)
    decoder_2 = decoder_block(decoder_3, CBAM(encoder_2[1]), f)
    decoder_1 = decoder_block(decoder_2, CBAM(encoder_1[1]), f)

    output = layers.Conv2DTranspose(1, (2,2), padding='same',kernel_initializer='he_normal', kernel_regularizer='l1')(decoder_1)
    output = CBAM(output)
    out_image = layers.Activation('sigmoid')(output)
    model = Model(inputs=inp, outputs = out_image)

    if not encoder_trainable:
        for layer in model.layers:
            if layer.name == 'Model_Neck_1':
                break
            layer.trainable = False

    return model

## build Generator
images_shape_ = (256,256,3)
generator = build_generator(images_shape_,f=20, encoder_trainable=True)
opt = Adamax(learning_rate=0.001)
generator.compile(optimizer=opt, loss=VGG_loss, metrics=['mae', psnr, BC_loss, ResNet50_loss, ssim_metric, 'mse', 'poisson'])


################ callbacks
# class PerformanceCallback_Plot(Callback):
#     def __init__(self, Flair, T1 , T2, Contrast, input_, save_path, curriculum=None):
#         self.Flair = Flair
#         self.T1 = T1
#         self.T2 = T2
#         self.contrast = Contrast
#         self.input_ = input_
#         self.Img_titles = ['T2', 'Flair', 'T1', 'Contrast Truth', 'Generated Image', 'Contrast Overlay']#'Equalized/Deconvolved \n Generated Image']
#         self.epoch = 0
#         self.save_path = save_path
#         self.curriculum = curriculum

#     def __call__(self, Flair, T1 , T2, Contrast, Target, save_path):
#         print('The Performance Callback Plot Class is already Initiated')

#     def on_epoch_end(self, epoch, logs=None):
#         self.int_ = np.random.randint(0, len(self.contrast))
#         self.epoch += 1
#         self.gen_img = equalize_adapthist(self.model.predict(np.expand_dims(self.input_[self.int_], 0), verbose = 0))
#         self.filtered_image = denoise_bilateral(np.squeeze(np.squeeze(self.gen_img, 0), -1), sigma_color=0.05, sigma_spatial=15)
#         self.image_list = [self.T2[self.int_], self.Flair[self.int_], self.T1[self.int_], self.contrast[self.int_], np.squeeze(self.gen_img, 0), self.filtered_image]
#         fig, axes = plt.subplots(2, 3, facecolor='lightsteelblue', figsize = (10,7))
#         if self.curriculum:
#             fig.suptitle('curriculum %d - Epoch no: %d \n '% (self.curriculum, epoch),
#                      fontsize = 18,
#                      color = 'darkgoldenrod')
#         else:
#             fig.suptitle('Epoch no: %d \n '% (epoch),
#                      fontsize = 18,
#                      color = 'darkgoldenrod')
#         axes = axes.ravel()
#         for i, ax in enumerate((axes)):
#             if self.Img_titles[i] == 'Contrast Overlay':
#                 ax.imshow(self.image_list[2], cmap='gray')
#                 ax.imshow(self.image_list[i], cmap='gray', alpha = 0.5)
#             else:
#                 ax.imshow(self.image_list[i], cmap='gray')
#             ax.axis('off')
#             ax.set_title(self.Img_titles[i], fontsize = 18, color='darkblue')
#         plt.tight_layout(pad=1.3)
#         if self.curriculum:
#             plt.savefig(self.save_path+'curriculum {} Epoch {}.png'.format(self.curriculum, self.epoch), dpi = 250)
#         else:
#             plt.savefig(self.save_path+'Epoch {}.png'.format(self.epoch), dpi = 250)
#         plt.show()

#         if epoch % 20 == 0:
#             if self.curriculum:
#                 self.model.save(self.save_path+'Generator curriculum {} epoch {}.h5'.format(self.curriculum, self.epoch))
#             else:
#                 self.model.save(self.save_path+'Generator epoch {}.h5'.format(self.epoch))


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.99, patience=5, min_lr=0.0001)
earlystoping = EarlyStopping('val_loss', patience=15)


def image_to_base64(image_array: np.ndarray) -> str:
    image_array = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    byte_data = buf.getvalue()
    return base64.b64encode(byte_data).decode('utf-8')


###### evaluate model Metrics
#history = generator.evaluate(input_ar, output_ar, batch_size=1, verbose=1)

def prepare_input(T1, T2, Flair) -> np.ndarray:   
    """
    Stack T1, T2, Flair into a 3-channel image of shape (1, 256, 256, 3)
    """
    stacked = np.stack([T1, T2, Flair], axis=-1)  # shape: (256, 256, 3)
    return np.expand_dims(stacked, axis=0)        # shape: (1, 256, 256, 3)

def predict_array(model, array_list):
    T1, T2, Flair = array_list
    

    
    # Generate predictions using the model
    predict_imgs = model.predict(np.expand_dims(T1, axis=0), verbose=1, batch_size=3)
    
    # Post-process each predicted image
    processed_images = []
    for id_ in range(len(predict_imgs)):
        gen_img = equalize_adapthist(predict_imgs[id_])  # Contrast enhancement
        filtered_image = denoise_bilateral(np.squeeze(gen_img, -1), sigma_color=0.05, sigma_spatial=15)  # Denoising
        processed_images.append(filtered_image)

    return processed_images


def download_file_on_startup():
    file_id = '1jAyygn2H9vZQLJ-s1Zw1pDOtjd0zeag-'
    url = f'https://drive.google.com/uc?id={file_id}'
    output_path = os.getcwd()+'/weight.h5'

    if not os.path.exists(output_path):
        print("Downloading file...")
        gdown.download(url, output_path, quiet=False)
        print(f"File downloaded to {output_path}")
        with h5py.File('weights.h5', 'r') as f:
            print("Top-level groups:", list(f.keys()))
            for layer_name in f.keys():
                print(f"Layer: {layer_name}")
                print("Weights:", list(f[layer_name].keys()))

        generator.load_weights('weights.h5', by_name=True, skip_mismatch=True)
        # generator.load_weights('weights.h5')
        print(generator.summary())
    else:
        print("File already exists. Skipping download.")



app = FastAPI()

@app.on_event("startup")
def startup_event():
    download_file_on_startup()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


def read_image(file: UploadFile) -> np.ndarray:
    """Read uploaded file and convert it to RGB numpy array of shape (256, 256, 3)"""
    image = Image.open(io.BytesIO(file.file.read())).convert('RGB').resize((256, 256))
    return np.array(image, dtype=np.float32) / 255.0  # normalize to [0, 1]


@app.post("/predict-contrast")
async def predict_contrast(
    T1: UploadFile = File(...),
    T2: UploadFile = File(...),
    Flair: UploadFile = File(...)
):
    
    # input_array = read_image(input_ar)
    T1 = read_image(T1)
    T2 = read_image(T2)
    Flair = read_image(Flair)

    prediction_list = [T1, T2, Flair]
    processed_images = predict_array(generator, prediction_list)

    # Convert the first predicted image to base64
    base64_image = image_to_base64(processed_images[0])
    
    return JSONResponse(content={"enhanced_image_base64": base64_image})

# Image Segmentation with U-Net Model

## I. Download & Manipulate Dataset

### 1. Download dataset


```python
!pip install -q git+https://github.com/tensorflow/examples.git
```

      Building wheel for tensorflow-examples (setup.py) ... [?25l[?25hdone
    


```python
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds

from IPython.display import clear_output
import matplotlib.pyplot as plt
```


```python
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
```

    [1mDownloading and preparing dataset oxford_iiit_pet/3.2.0 (download: 773.52 MiB, generated: 774.69 MiB, total: 1.51 GiB) to /root/tensorflow_datasets/oxford_iiit_pet/3.2.0...[0m
    


    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Dl Completed...', max=1.0, style=Progre…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Dl Size...', max=1.0, style=ProgressSty…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Extraction completed...', max=1.0, styl…


    
    
    
    
    
    
    


    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    Shuffling and writing examples to /root/tensorflow_datasets/oxford_iiit_pet/3.2.0.incomplete7H5RWL/oxford_iiit_pet-train.tfrecord
    


    HBox(children=(FloatProgress(value=0.0, max=3680.0), HTML(value='')))


    


    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    Shuffling and writing examples to /root/tensorflow_datasets/oxford_iiit_pet/3.2.0.incomplete7H5RWL/oxford_iiit_pet-test.tfrecord
    


    HBox(children=(FloatProgress(value=0.0, max=3669.0), HTML(value='')))


    [1mDataset oxford_iiit_pet downloaded and prepared to /root/tensorflow_datasets/oxford_iiit_pet/3.2.0. Subsequent calls will reuse this data.[0m
    


```python
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
```

### 2. Load dataset


```python
@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], IMAGE_SIZE)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], IMAGE_SIZE)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], IMAGE_SIZE)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], IMAGE_SIZE)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask
```

### 3. Separate the train-test dataset


```python
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 128
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
```


```python
train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)
```

## II. Dataset Ultility

### 1. Display the overall images


```python
import numpy as np
def display(display_list, visible_mask = np.array([True, True, True, True])):
    plt.figure(figsize=(20, 20))
    title = ["Input Image", "True Mask", "Predicted Mask", "Remove Background"]
    title = tf.boolean_mask(title, visible_mask).numpy().astype('str')
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
```

### Mask convert


```python
def convert_mask(t, target = 1.0):
    return tf.math.abs( 
        tf.math.subtract(
            t, 
            tf.convert_to_tensor(target, dtype=tf.float32) 
        ) 
    )

def remove_background(image, mask):
    msk = convert_mask(mask)
    return tf.math.multiply(image, msk)

```

### 3. Some example from train dataset


```python
for image, mask in train.take(5):
    sample_image, sample_mask = image, mask
    result = remove_background(image, mask)
    display([sample_image, sample_mask, result], [True, True, False, True])
```


![png](output_18_0.png)



![png](output_18_1.png)



![png](output_18_2.png)



![png](output_18_3.png)



![png](output_18_4.png)


## III. Construct model

### 1. U-Net model use MobileNetV2 Encode-Decode


```python
from tensorflow.keras.applications import MobileNetV2 as Pretrained_Model
from tensorflow.keras.layers import *

def unet_model_1(output_channels):

    base_model = Pretrained_Model(
        input_shape=INPUT_SHAPE, 
        include_top=False
    )

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_expand_relu',  # 4x4
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(1024, 3),  # 4x4 -> 8x8
        pix2pix.upsample(512, 3),  # 8x8 -> 16x16
        pix2pix.upsample(256, 3),  # 16x16 -> 32x32
        pix2pix.upsample(128, 3),   # 32x32 -> 64x64
    ]

    inputs = Input(shape=INPUT_SHAPE)
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same'
    )  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
```

### 2. U-Net model use MobileNetV3Large Encode-Decode


```python
from tensorflow.keras.applications import MobileNetV3Large as Pretrained_Model_2
from tensorflow.keras.layers import *

def unet_model_2(output_channels):

    base_model = Pretrained_Model_2(
        input_shape=INPUT_SHAPE, 
        include_top=False
    )

    for index, layer in enumerate(base_model.layers):
        layer._name = 'mylayer_' + str(index)

    # Use the activations of these layers
    layer_names = [
        'mylayer_16',   # 64x64
        'mylayer_28',   # 32x32
        'mylayer_89',   # 16x16
        'mylayer_178',  # 8x8
        'mylayer_273'   # 4x4
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(1024, 3),  # 4x4 -> 8x8
        pix2pix.upsample(512, 3),  # 8x8 -> 16x16
        pix2pix.upsample(256, 3),  # 16x16 -> 32x32
        pix2pix.upsample(128, 3),   # 32x32 -> 64x64
    ]

    inputs = Input(shape=INPUT_SHAPE)
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same'
    )  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
```

### 3. Self-made Unet


```python
from tensorflow.keras.layers import *

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1, numberOfLayer = 2):
    input_layer = x
    for i in range(numberOfLayer):
        input_layer = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(input_layer)
    return input_layer

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1, numberOfLayer = 2):
    c = bottleneck(x, filters, kernel_size, padding, strides, numberOfLayer)
    p = MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1, numberOfLayer = 2):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = bottleneck(concat, filters, kernel_size, padding, strides, numberOfLayer)
    return c
```


```python
def unet_model_3(channels):
    f = [32, 64, 128, 256, 512, 1024]
    numOfLayer = 4
    inputs = Input(INPUT_SHAPE)
    
    conv = []
    pool = inputs
    for i in range(len(f) - 1):
        c, pool = down_block(pool, f[1], numberOfLayer=numOfLayer)
        conv.append(c)

    # c1, p1 = down_block(p0, f[0]) #128 -> 64
    # c2, p2 = down_block(p1, f[1]) #64 -> 32
    # c3, p3 = down_block(p2, f[2]) #32 -> 16
    # c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(pool, f[-1])
    up = bn
    
    for i in range(len(f) - 2, 0, -1):
        up = up_block(up, conv.pop(), f[i], numberOfLayer=numOfLayer)

    # u1 = up_block(bn, c4, f[3]) #8 -> 16
    # u2 = up_block(u1, c3, f[2]) #16 -> 32
    # u3 = up_block(u2, c2, f[1]) #32 -> 64
    # u4 = up_block(u3, c1, f[0]) #64 -> 128

    # outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(up)
    outputs = Conv2DTranspose(channels, 3, strides=2, padding='same', activation='softmax')(up)

    model = tf.keras.models.Model(inputs, outputs)
    return model
```

## IV. Let's get started

### 0. Model Ultility


```python
class ModelManager(object):

    def __init__(self):
        self._ModelRegistry_ = {}

    def is_key_exist(self, key):
        return key in self._ModelRegistry_.keys()

    def is_model_exist(self, model):
        return model in self._ModelRegistry_.values()

    def register_model(self, key, model):
        if self.is_key_exist(key):
            if self.__ModelRegistry_[key] == model:
                return model
            print("Key exist in registry and registered for another model!")
            return model

        if self.is_model_exist(model):
            if self._ModelRegistry_.index(model) == key:
                return model
            print('Model is already registred with index : ' + str(self._ModelRegistry_.index(model)))
            return model

        print('Register done!')
        self._ModelRegistry_[key] = model
        return model

    def get_model(self, key):
        return self._ModelRegistry_[key]
```

### 1. Register all models


```python
manager = ModelManager()

manager.register_model('unet_mobilenetv2', unet_model_1(3)),
manager.register_model('unet_mobilenetv3', unet_model_2(3)),
manager.register_model('unet_diy', unet_model_3(3))
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
    9412608/9406464 [==============================] - 0s 0us/step
    Register done!
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/weights_mobilenet_v3_large_224_1.0_float_no_top.h5
    17612800/17605208 [==============================] - 0s 0us/step
    Register done!
    Register done!
    




    <tensorflow.python.keras.engine.functional.Functional at 0x7efc34711f98>



### 2. Brief information of Model 


```python
#@title Select a model to try { run: "auto", vertical-output: true, display-mode: "form" }
model_name = "unet_mobilenetv2" #@param ["unet_mobilenetv2", "unet_mobilenetv3", "unet_diy"] {allow-input: true}
print(model_name + " is choose!")
```

    unet_mobilenetv2 is choose!
    


```python
model = manager.get_model(model_name)
```


```python
model.summary()
```

    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            [(None, 224, 224, 3) 0                                            
    __________________________________________________________________________________________________
    model (Functional)              [(None, 112, 112, 96 1522304     input_2[0][0]                    
    __________________________________________________________________________________________________
    sequential (Sequential)         (None, 14, 14, 1024) 8851456     model[0][4]                      
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 14, 14, 1600) 0           sequential[0][0]                 
                                                                     model[0][3]                      
    __________________________________________________________________________________________________
    sequential_1 (Sequential)       (None, 28, 28, 512)  7374848     concatenate[0][0]                
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 28, 28, 704)  0           sequential_1[0][0]               
                                                                     model[0][2]                      
    __________________________________________________________________________________________________
    sequential_2 (Sequential)       (None, 56, 56, 256)  1623040     concatenate_1[0][0]              
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, 56, 56, 400)  0           sequential_2[0][0]               
                                                                     model[0][1]                      
    __________________________________________________________________________________________________
    sequential_3 (Sequential)       (None, 112, 112, 128 461312      concatenate_2[0][0]              
    __________________________________________________________________________________________________
    concatenate_3 (Concatenate)     (None, 112, 112, 224 0           sequential_3[0][0]               
                                                                     model[0][0]                      
    __________________________________________________________________________________________________
    conv2d_transpose_4 (Conv2DTrans (None, 224, 224, 3)  6051        concatenate_3[0][0]              
    ==================================================================================================
    Total params: 19,839,011
    Trainable params: 18,312,867
    Non-trainable params: 1,526,144
    __________________________________________________________________________________________________
    


```python
tf.keras.utils.plot_model(model, show_shapes=True)
```




![png](output_36_0.png)



### 3. Training Model


```python
OUTPUT_CHANNELS = 3
EPOCHS = 50
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

optimizer = tf.keras.optimizers.Adam(
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.03,
        decay_steps=10000,
        decay_rate=0.95,
        staircase=True
    )
)

loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

metrics = ['accuracy']
```


```python
model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metrics
)

history = model.fit(
    train_dataset, epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    validation_data=test_dataset,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ]
)
```

    Epoch 1/50
    28/28 [==============================] - 61s 2s/step - loss: 4.0434 - accuracy: 0.4565 - val_loss: 16.5655 - val_accuracy: 0.5628
    Epoch 2/50
    28/28 [==============================] - 48s 2s/step - loss: 1.0608 - accuracy: 0.5953 - val_loss: 1.5554 - val_accuracy: 0.6071
    Epoch 3/50
    28/28 [==============================] - 46s 2s/step - loss: 0.4784 - accuracy: 0.8213 - val_loss: 4.1752 - val_accuracy: 0.6070
    Epoch 4/50
    28/28 [==============================] - 49s 2s/step - loss: 0.3256 - accuracy: 0.8676 - val_loss: 3.3945 - val_accuracy: 0.6113
    Epoch 5/50
    28/28 [==============================] - 49s 2s/step - loss: 0.2898 - accuracy: 0.8803 - val_loss: 1.9399 - val_accuracy: 0.6423
    Epoch 6/50
    28/28 [==============================] - 49s 2s/step - loss: 0.2631 - accuracy: 0.8908 - val_loss: 1.4994 - val_accuracy: 0.6549
    Epoch 7/50
    28/28 [==============================] - 49s 2s/step - loss: 0.2570 - accuracy: 0.8933 - val_loss: 0.6933 - val_accuracy: 0.7707
    Epoch 8/50
    28/28 [==============================] - 49s 2s/step - loss: 0.2491 - accuracy: 0.8964 - val_loss: 0.4914 - val_accuracy: 0.8232
    Epoch 9/50
    28/28 [==============================] - 50s 2s/step - loss: 0.2372 - accuracy: 0.9004 - val_loss: 0.4374 - val_accuracy: 0.8355
    Epoch 10/50
    28/28 [==============================] - 50s 2s/step - loss: 0.2364 - accuracy: 0.9008 - val_loss: 0.3297 - val_accuracy: 0.8677
    Epoch 11/50
    28/28 [==============================] - 50s 2s/step - loss: 0.2391 - accuracy: 0.8996 - val_loss: 0.2910 - val_accuracy: 0.8820
    Epoch 12/50
    28/28 [==============================] - 49s 2s/step - loss: 0.2279 - accuracy: 0.9036 - val_loss: 0.2691 - val_accuracy: 0.8902
    Epoch 13/50
    28/28 [==============================] - 49s 2s/step - loss: 0.2216 - accuracy: 0.9056 - val_loss: 0.2585 - val_accuracy: 0.8959
    Epoch 14/50
    28/28 [==============================] - 50s 2s/step - loss: 0.2168 - accuracy: 0.9072 - val_loss: 0.2517 - val_accuracy: 0.8996
    Epoch 15/50
    28/28 [==============================] - 50s 2s/step - loss: 0.2172 - accuracy: 0.9076 - val_loss: 0.2509 - val_accuracy: 0.8994
    Epoch 16/50
    28/28 [==============================] - 50s 2s/step - loss: 0.2092 - accuracy: 0.9101 - val_loss: 0.2558 - val_accuracy: 0.8980
    Epoch 17/50
    28/28 [==============================] - 50s 2s/step - loss: 0.2053 - accuracy: 0.9116 - val_loss: 0.2681 - val_accuracy: 0.8958
    Epoch 18/50
    28/28 [==============================] - 50s 2s/step - loss: 0.2089 - accuracy: 0.9105 - val_loss: 0.2569 - val_accuracy: 0.9003
    Epoch 19/50
    28/28 [==============================] - 50s 2s/step - loss: 0.1971 - accuracy: 0.9144 - val_loss: 0.2590 - val_accuracy: 0.9002
    Epoch 20/50
    28/28 [==============================] - 49s 2s/step - loss: 0.1942 - accuracy: 0.9155 - val_loss: 0.2718 - val_accuracy: 0.8968
    

### 4. Model Evaluation


```python
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, max(loss)])
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure()
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.show()
```


![png](output_41_0.png)



![png](output_41_1.png)



```python
model.evaluate(
    test_dataset, 
    batch_size=STEPS_PER_EPOCH
)
```

    29/29 [==============================] - 23s 790ms/step - loss: 0.2690 - accuracy: 0.8971
    




    [0.2690197825431824, 0.8970551490783691]



## V. Application in real example


```python
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]
```


```python
def show_predictions(model, dataset=None, num=1):
   
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = create_mask(model.predict(image))
            result = remove_background(
                image[0],
                tf.cast(pred_mask, tf.float32)
            )
            display([image[0], mask[0], pred_mask, result])
    else:
        pred_mask =  create_mask(model.predict(sample_image[tf.newaxis, ...]))
        # result = remove_background(sample_image, pred_mask)
        display([sample_image, sample_mask, pred_mask])
```

### 1. From test dataset


```python
show_predictions(model, test_dataset, 10)
```


![png](output_47_0.png)



![png](output_47_1.png)



![png](output_47_2.png)



![png](output_47_3.png)



![png](output_47_4.png)



![png](output_47_5.png)



![png](output_47_6.png)



![png](output_47_7.png)



![png](output_47_8.png)



![png](output_47_9.png)


### 2. Let try your example 


```python
import matplotlib.pyplot as plt
import cv2 as cv
from google.colab import files
uploaded = files.upload()

for name, data in uploaded.items():
  with open(name, 'wb') as f:
    f.write(data)
    print ('saved file', name)

img = cv.imread(name)
img_cvt=cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img_cvt)
plt.show()
```



<input type="file" id="files-2219a796-1ed9-49f7-ac1f-1a3403c04e19" name="files[]" multiple disabled
   style="border:none" />
<output id="result-2219a796-1ed9-49f7-ac1f-1a3403c04e19">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving download.jpg to download.jpg
    saved file download.jpg
    


![png](output_49_2.png)



```python
resize_img = cv.resize(img, IMAGE_SIZE)

reshape_img = np.reshape(resize_img , (1, resize_img.shape[0], resize_img.shape[1], resize_img.shape[2]))

pred_mask = create_mask(model.predict(reshape_img))

result = remove_background(
    resize_img,
    tf.cast(pred_mask, tf.float32)
)

display([cv.cvtColor(resize_img, cv.COLOR_BGR2RGB), pred_mask, result], [True, False, True, True])
```


![png](output_50_0.png)


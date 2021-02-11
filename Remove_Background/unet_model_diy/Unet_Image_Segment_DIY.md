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
    numOfLayer = 2
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

# manager.register_model('unet_mobilenetv2', unet_model_1(3)),
# manager.register_model('unet_mobilenetv3', unet_model_2(3)),
manager.register_model('unet_diy', unet_model_3(3))
```

    Register done!
    




    <tensorflow.python.keras.engine.functional.Functional at 0x7f7ea8a6f550>



### 2. Brief information of Model 


```python
#@title Select a model to try { run: "auto", vertical-output: true, display-mode: "form" }
model_name = "unet_diy" #@param ["unet_mobilenetv2", "unet_mobilenetv3", "unet_diy"] {allow-input: true}
print(model_name + " is choose!")
```

    unet_diy is choose!
    


```python
model = manager.get_model(model_name)
```


```python
model.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_3 (InputLayer)            [(None, 224, 224, 3) 0                                            
    __________________________________________________________________________________________________
    conv2d_30 (Conv2D)              (None, 224, 224, 64) 1792        input_3[0][0]                    
    __________________________________________________________________________________________________
    conv2d_31 (Conv2D)              (None, 224, 224, 64) 36928       conv2d_30[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_13 (MaxPooling2D) (None, 112, 112, 64) 0           conv2d_31[0][0]                  
    __________________________________________________________________________________________________
    conv2d_32 (Conv2D)              (None, 112, 112, 64) 36928       max_pooling2d_13[0][0]           
    __________________________________________________________________________________________________
    conv2d_33 (Conv2D)              (None, 112, 112, 64) 36928       conv2d_32[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_14 (MaxPooling2D) (None, 56, 56, 64)   0           conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    conv2d_34 (Conv2D)              (None, 56, 56, 64)   36928       max_pooling2d_14[0][0]           
    __________________________________________________________________________________________________
    conv2d_35 (Conv2D)              (None, 56, 56, 64)   36928       conv2d_34[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_15 (MaxPooling2D) (None, 28, 28, 64)   0           conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    conv2d_36 (Conv2D)              (None, 28, 28, 64)   36928       max_pooling2d_15[0][0]           
    __________________________________________________________________________________________________
    conv2d_37 (Conv2D)              (None, 28, 28, 64)   36928       conv2d_36[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_16 (MaxPooling2D) (None, 14, 14, 64)   0           conv2d_37[0][0]                  
    __________________________________________________________________________________________________
    conv2d_38 (Conv2D)              (None, 14, 14, 64)   36928       max_pooling2d_16[0][0]           
    __________________________________________________________________________________________________
    conv2d_39 (Conv2D)              (None, 14, 14, 64)   36928       conv2d_38[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_17 (MaxPooling2D) (None, 7, 7, 64)     0           conv2d_39[0][0]                  
    __________________________________________________________________________________________________
    conv2d_40 (Conv2D)              (None, 7, 7, 1024)   590848      max_pooling2d_17[0][0]           
    __________________________________________________________________________________________________
    conv2d_41 (Conv2D)              (None, 7, 7, 1024)   9438208     conv2d_40[0][0]                  
    __________________________________________________________________________________________________
    up_sampling2d_2 (UpSampling2D)  (None, 14, 14, 1024) 0           conv2d_41[0][0]                  
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, 14, 14, 1088) 0           up_sampling2d_2[0][0]            
                                                                     conv2d_39[0][0]                  
    __________________________________________________________________________________________________
    conv2d_42 (Conv2D)              (None, 14, 14, 512)  5014016     concatenate_2[0][0]              
    __________________________________________________________________________________________________
    conv2d_43 (Conv2D)              (None, 14, 14, 512)  2359808     conv2d_42[0][0]                  
    __________________________________________________________________________________________________
    up_sampling2d_3 (UpSampling2D)  (None, 28, 28, 512)  0           conv2d_43[0][0]                  
    __________________________________________________________________________________________________
    concatenate_3 (Concatenate)     (None, 28, 28, 576)  0           up_sampling2d_3[0][0]            
                                                                     conv2d_37[0][0]                  
    __________________________________________________________________________________________________
    conv2d_44 (Conv2D)              (None, 28, 28, 256)  1327360     concatenate_3[0][0]              
    __________________________________________________________________________________________________
    conv2d_45 (Conv2D)              (None, 28, 28, 256)  590080      conv2d_44[0][0]                  
    __________________________________________________________________________________________________
    up_sampling2d_4 (UpSampling2D)  (None, 56, 56, 256)  0           conv2d_45[0][0]                  
    __________________________________________________________________________________________________
    concatenate_4 (Concatenate)     (None, 56, 56, 320)  0           up_sampling2d_4[0][0]            
                                                                     conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    conv2d_46 (Conv2D)              (None, 56, 56, 128)  368768      concatenate_4[0][0]              
    __________________________________________________________________________________________________
    conv2d_47 (Conv2D)              (None, 56, 56, 128)  147584      conv2d_46[0][0]                  
    __________________________________________________________________________________________________
    up_sampling2d_5 (UpSampling2D)  (None, 112, 112, 128 0           conv2d_47[0][0]                  
    __________________________________________________________________________________________________
    concatenate_5 (Concatenate)     (None, 112, 112, 192 0           up_sampling2d_5[0][0]            
                                                                     conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    conv2d_48 (Conv2D)              (None, 112, 112, 64) 110656      concatenate_5[0][0]              
    __________________________________________________________________________________________________
    conv2d_49 (Conv2D)              (None, 112, 112, 64) 36928       conv2d_48[0][0]                  
    __________________________________________________________________________________________________
    conv2d_transpose (Conv2DTranspo (None, 224, 224, 3)  1731        conv2d_49[0][0]                  
    ==================================================================================================
    Total params: 20,320,131
    Trainable params: 20,320,131
    Non-trainable params: 0
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
        initial_learning_rate = 0.0001,
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
    28/28 [==============================] - 72s 2s/step - loss: 1.0769 - accuracy: 0.3946 - val_loss: 0.9197 - val_accuracy: 0.5658
    Epoch 2/50
    28/28 [==============================] - 61s 2s/step - loss: 0.8471 - accuracy: 0.5816 - val_loss: 0.8166 - val_accuracy: 0.5948
    Epoch 3/50
    28/28 [==============================] - 53s 2s/step - loss: 0.7801 - accuracy: 0.6339 - val_loss: 0.7710 - val_accuracy: 0.6867
    Epoch 4/50
    28/28 [==============================] - 54s 2s/step - loss: 0.7385 - accuracy: 0.6981 - val_loss: 0.7433 - val_accuracy: 0.6968
    Epoch 5/50
    28/28 [==============================] - 53s 2s/step - loss: 0.7118 - accuracy: 0.7071 - val_loss: 0.7010 - val_accuracy: 0.7120
    Epoch 6/50
    28/28 [==============================] - 54s 2s/step - loss: 0.6604 - accuracy: 0.7304 - val_loss: 0.6628 - val_accuracy: 0.7280
    Epoch 7/50
    28/28 [==============================] - 54s 2s/step - loss: 0.6302 - accuracy: 0.7409 - val_loss: 0.6121 - val_accuracy: 0.7474
    Epoch 8/50
    28/28 [==============================] - 54s 2s/step - loss: 0.6121 - accuracy: 0.7496 - val_loss: 0.6264 - val_accuracy: 0.7382
    Epoch 9/50
    28/28 [==============================] - 54s 2s/step - loss: 0.5929 - accuracy: 0.7548 - val_loss: 0.5796 - val_accuracy: 0.7559
    Epoch 10/50
    28/28 [==============================] - 54s 2s/step - loss: 0.5563 - accuracy: 0.7683 - val_loss: 0.5425 - val_accuracy: 0.7714
    Epoch 11/50
    28/28 [==============================] - 54s 2s/step - loss: 0.5292 - accuracy: 0.7784 - val_loss: 0.5722 - val_accuracy: 0.7595
    Epoch 12/50
    28/28 [==============================] - 54s 2s/step - loss: 0.5292 - accuracy: 0.7785 - val_loss: 0.5040 - val_accuracy: 0.7878
    Epoch 13/50
    28/28 [==============================] - 54s 2s/step - loss: 0.4901 - accuracy: 0.7949 - val_loss: 0.4861 - val_accuracy: 0.7972
    Epoch 14/50
    28/28 [==============================] - 54s 2s/step - loss: 0.4741 - accuracy: 0.8034 - val_loss: 0.4860 - val_accuracy: 0.7971
    Epoch 15/50
    28/28 [==============================] - 54s 2s/step - loss: 0.4584 - accuracy: 0.8093 - val_loss: 0.4619 - val_accuracy: 0.8090
    Epoch 16/50
    28/28 [==============================] - 54s 2s/step - loss: 0.4305 - accuracy: 0.8219 - val_loss: 0.4570 - val_accuracy: 0.8110
    Epoch 17/50
    28/28 [==============================] - 54s 2s/step - loss: 0.4258 - accuracy: 0.8220 - val_loss: 0.4425 - val_accuracy: 0.8163
    Epoch 18/50
    28/28 [==============================] - 55s 2s/step - loss: 0.4086 - accuracy: 0.8313 - val_loss: 0.4649 - val_accuracy: 0.8080
    Epoch 19/50
    28/28 [==============================] - 54s 2s/step - loss: 0.3956 - accuracy: 0.8359 - val_loss: 0.4266 - val_accuracy: 0.8274
    Epoch 20/50
    28/28 [==============================] - 54s 2s/step - loss: 0.3853 - accuracy: 0.8406 - val_loss: 0.4288 - val_accuracy: 0.8268
    Epoch 21/50
    28/28 [==============================] - 54s 2s/step - loss: 0.3845 - accuracy: 0.8408 - val_loss: 0.4198 - val_accuracy: 0.8295
    Epoch 22/50
    28/28 [==============================] - 54s 2s/step - loss: 0.3669 - accuracy: 0.8483 - val_loss: 0.4091 - val_accuracy: 0.8341
    Epoch 23/50
    28/28 [==============================] - 55s 2s/step - loss: 0.3468 - accuracy: 0.8554 - val_loss: 0.4258 - val_accuracy: 0.8332
    Epoch 24/50
    28/28 [==============================] - 54s 2s/step - loss: 0.3356 - accuracy: 0.8600 - val_loss: 0.4178 - val_accuracy: 0.8360
    Epoch 25/50
    28/28 [==============================] - 55s 2s/step - loss: 0.3302 - accuracy: 0.8621 - val_loss: 0.4189 - val_accuracy: 0.8357
    Epoch 26/50
    28/28 [==============================] - 54s 2s/step - loss: 0.3123 - accuracy: 0.8686 - val_loss: 0.4360 - val_accuracy: 0.8312
    Epoch 27/50
    28/28 [==============================] - 54s 2s/step - loss: 0.3093 - accuracy: 0.8704 - val_loss: 0.4641 - val_accuracy: 0.8330
    

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

    29/29 [==============================] - 17s 582ms/step - loss: 0.4625 - accuracy: 0.8346
    




    [0.4624691903591156, 0.8345950841903687]



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



<input type="file" id="files-cb1000bc-92ec-482e-98a2-9a2d873c224a" name="files[]" multiple disabled
   style="border:none" />
<output id="result-cb1000bc-92ec-482e-98a2-9a2d873c224a">
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


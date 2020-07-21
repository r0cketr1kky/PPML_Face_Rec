# VGG-18_LIGHT


VGG-16_Light Weights : https://drive.google.com/file/d/19Ck7kyhEwRSoEZzvRcP11ftQ3PIaKlnc/view?usp=sharing

VGG-16_Light architecture : 
```
model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), \
                               padding='same', \
                               activation='relu', \
                               input_shape=input_shape),
        tf.keras.layers.AveragePooling2D((2,2), strides=(2,2)),
        tf.keras.layers.Conv2D(128, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.AveragePooling2D((2,2), strides=(2,2)),
        tf.keras.layers.Conv2D(256, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.Conv2D(256, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.AveragePooling2D((2,2), strides=(2,2)),
        tf.keras.layers.Conv2D(512, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.AveragePooling2D((2,2), strides=(2,2)),
        tf.keras.layers.Conv2D(512, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.AveragePooling2D((2,2), strides=(2,2)),
        tf.keras.layers.Conv2D(10, (7,7), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D()
])
```

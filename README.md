# PPML_Face_Rec


Modified VGG-16 Weights : https://drive.google.com/file/d/1Tn8Y7ZB1AyU3OqTysJfcAQptieS6y2z8/view?usp=sharing

Modified VGG-16 architecture : 

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), \
                               padding='same', \
                               activation='relu', \
                               batch_input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.AveragePooling2D((2,2), strides=(2,2)),
        tf.keras.layers.Conv2D(128, (3,3), \
                               padding='same', \
                               activation='relu'),
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
        tf.keras.layers.Conv2D(256, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.AveragePooling2D((2,2), strides=(2,2)),
        tf.keras.layers.Conv2D(512, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.Conv2D(512, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.Conv2D(512, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.AveragePooling2D((2,2), strides=(2,2)),
        tf.keras.layers.Conv2D(512, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.Conv2D(512, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.Conv2D(512, (3,3), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.AveragePooling2D((2,2), strides=(2,2)),
        tf.keras.layers.Conv2D(10, (7,7), \
                               padding='same', \
                               activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D()
])

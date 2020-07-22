import tf_encrypted as tfe
import tensorflow as tf
import numpy as np

input_shape = (1, 224, 224, 3)

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), \
                               padding='same', \
                               activation='relu', \
                               batch_input_shape=input_shape),
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

pre_trained_weights = 'my_model6.h5'
model.load_weights(pre_trained_weights)

from collections import OrderedDict

players = OrderedDict([
    ('server0', 'localhost:4000'),
    ('server1', 'localhost:4001'),
    ('server2', 'localhost:4002'),
])

config = tfe.RemoteConfig(players)
config.save('/tmp/tfe.config')

tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())

tfe_model = tfe.keras.models.clone_model(model)

for player_name in players.keys():
    print("python -m tf_encrypted.player --config /tmp/tfe.config {}".format(player_name))


q_input_shape = (1, 224, 224, 3)
q_output_shape = (1, 10)


server = tfe.serving.QueueServer(input_shape=q_input_shape, output_shape=q_output_shape, computation_fn=tfe_model)

import tf_encrypted.keras.backend as KE
sess = tfe.Session(config=config)
#sess = KE.get_session()

request_ix = 1

def step_fn():
    global request_ix
    print("Served encrypted prediction {i} to client.".format(i=request_ix))
    request_ix += 1


server.run(sess, num_steps=1, step_fn=step_fn)

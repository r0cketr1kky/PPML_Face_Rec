import tf_encrypted as tfe
import tensorflow as tf

config = tfe.RemoteConfig.load("/tmp/tfe.config")

tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())

input_shape = (1, 224, 224, 3)
output_shape = (1, 10)

client = tfe.serving.QueueClient(input_shape=input_shape, output_shape=output_shape)

sess = tfe.Session(config=config)
fpath = 'subject06_centerlight-0000.jpg'

from matplotlib import pyplot
from PIL import Image
from numpy import asarray
import tensorflow as tf
import tf_encrypted as tfe 
import numpy as np
import time

def extract_face(filename, required_size=(224, 224, 3)):
    
    pixels = Image.open(filename).convert('RGB')
    image = np.asarray(pixels)
    image = np.resize(image, required_size)
    print(image.shape)

    return image

def get_embeddings(filenames):
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    print(samples.shape)
    samples /= 255
    #print(samples)

    starttime = time.time()
    
    res = client.run(sess, samples)
    print(res)
    
    predicted_label = np.argmax(res)
    print(predicted_label)
    
    print("Private prediction time is - ")
    print(time.time() - starttime)
    return

embeddings = get_embeddings([fpath])
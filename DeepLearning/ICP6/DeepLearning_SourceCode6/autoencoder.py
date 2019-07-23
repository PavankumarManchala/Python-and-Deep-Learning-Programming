from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
from time import time
import matplotlib.pyplot as plt

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(256, activation='relu')(input_img)
encode2 = Dense(64,activation='relu')(encoded)
encode1 = Dense(32,activation='relu')(encode2)
# "decoded" is the lossy reconstruction of the input
decode1 = Dense(32, activation='relu')(encode1)
decode2 = Dense(64, activation='relu')(decode1)
decoded = Dense(784, activation='sigmoid')(decode2)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
# encoder = Model(input_img, encode1)
# # create a placeholder for an encoded (32-dimensional) input
# encoded_input = Input(shape=(encoding_dim,))
# # retrieve the last layer of the autoencoder model
# decoder_layer = autoencoder.layers[-1]
# # create the decoder model
# decoder = Model(encoded_input, decoder_layer(encoded_input))
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])
from keras.datasets import mnist, fashion_mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
tb = TensorBoard(log_dir="logs/{}".format(time()))
ae = autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tb])

x_trans = x_test[40][np.newaxis]
prediction = autoencoder.predict(x_trans)

plt.imshow(x_test[40].reshape(28, 28), cmap='gray')
plt.show()
plt.imshow(prediction.reshape(28, 28), cmap='gray')
plt.show()

plt.plot(ae.history['acc'])
plt.plot(ae.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

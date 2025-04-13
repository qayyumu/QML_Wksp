import pennylane as qml
import tensorflow as tf
from pennylane import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
from os import path
import warnings
import logging

# Configure TensorFlow logging
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow info and warning messages

# Only suppress specific warnings that are known to be safe to ignore
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress future warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings
# Keep other warnings visible for debugging and improvement

n_qubits = 2 # Number of qubits should be the same as number of features, max number = 25
LABELplot=""
blocks = 6 #number of blocks (AngleEmbedding and StronglyEntanglingLayers is one block )
layers = 1  #layers per block (multiple "layers" of StronglyEntanglingLayers per block )
learning_rate = 0.02 #Learning rate for optimizer
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999) #tf.keras.optimizers.SGD(learning_rate=learning_rate) #Select optimizer

epochsH = 10 # Hybrid training epochs
batch_size = 16 # Batch size
test_size = 0.02 # Choose train-test split ratio
np.random.seed(42)


#  Generate Data
N = 1200 #No of points to generate
noise = 0.01 #add noise, for the makemoons dataset only


def squares(samples):
    data=[]
    Xvals, yvals = [], []
    dim=2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        if x[0] < 0 and x[1] < 0: y = 0
        if x[0] < 0 and x[1] > 0: y = 1
        if x[0] > 0 and x[1] < 0: y = 1
        if x[0] > 0 and x[1] > 0: y = 0        
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals), np.array(yvals)  
    return data, None

def wavy_lines2(samples, freq = 1):
    Xvals, yvals = [], []
    def fun1(s):
        return s + np.sin(freq * np.pi * s)
    
    def fun2(s):
        return -s + np.sin(freq * np.pi * s)
    data=[]
    dim=2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        if x[1] < fun1(x[0]) and x[1] < fun2(x[0]): y = 0
        if x[1] < fun1(x[0]) and x[1] > fun2(x[0]): y = 1
        if x[1] > fun1(x[0]) and x[1] < fun2(x[0]): y = 1
        if x[1] > fun1(x[0]) and x[1] > fun2(x[0]): y = 1        
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals), np.array(yvals)
    return data, freq    

    
def plot_data(x, y, fig=None, ax=None):
    """
    Plot data with red/blue values for a binary classification.

    Args:
        x (array[tuple]): array of data points as tuples
        y (array[int]): array of data points as tuples
    """
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    reds = y == 0
    blues = y == 1
    ax.scatter(x[reds, 0], x[reds, 1], c="red", s=20, edgecolor="k")
    ax.scatter(x[blues, 0], x[blues, 1], c="blue", s=20, edgecolor="k")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")   
    
    
    #X, y = circle(N)
#X, y = squares(N)
X, y = wavy_lines2(N)


y = pd.DataFrame.from_dict(y)
y = y.iloc[:, :]
y = y[0].apply(lambda x: 1 if x <= 0 else 0)
y = y.to_numpy()
#y_hot = tf.keras.utils.to_categorical(y, num_classes=2)  # one-hot encoded labels
#Normalize from 0 to pi
from sklearn.preprocessing import StandardScaler , minmax_scale
X = minmax_scale(X, feature_range=(0, np.pi))
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], c="r", marker="o", edgecolors="k")
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], c="b", marker="o", edgecolors="k")
plt.title("Dataset")
plt.axis('off')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)


# Define quantum node
dev = qml.device("default.qubit.tf", wires=n_qubits) # Run the model in classical CPU

@qml.qnode(dev, interface="tf", diff_method="backprop")
def qnode(inputs, weights):
    for i in range(blocks):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.StronglyEntanglingLayers(weights[i], wires=range(n_qubits)) #STRONGLY ENTANGLING LAYERS
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weights_shape = (blocks, layers, n_qubits, 3) # Uncomment for Strongly entangling layers

tf.keras.backend.set_floatx("float64")
weight_shapes = {"weights": weights_shape}

# Create the Hybrid model
#------------ classical Master layer ------------
clayerM = tf.keras.layers.Dense(X_train.shape[1], activation="relu") 
#------------ Quantum layer. It consists of the quantum node as defined before ------------
qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, n_qubits)
#------------ Classical Decision layer ------------
clayerD = tf.keras.layers.Dense(1, activation="sigmoid")
#--------------------------------------------------

inputs = tf.constant(np.random.random((batch_size, n_qubits)))

# Include classical and quantum layers 
modelh = tf.keras.models.Sequential([clayerM,qlayer,clayerD])


modelh.compile(opt, loss='binary_crossentropy',
               metrics=[AUC(name = 'auc')])#'sparse_categorical_accuracy','categorical_accuracy','binary_accuracy', 'accuracy'])

modelh.build(input_shape=X_train.shape)

historyh = modelh.fit(X_train, y_train,
                      #validation_split = 0.1,    
                      validation_data=(X_test, y_test),
                      epochs=epochsH,
                      batch_size=batch_size,
                     shuffle=True)

modelh.summary()

# Define quantum node
dev = qml.device("default.qubit.tf", wires=n_qubits) # Run the model in classical CPU

@qml.qnode(dev, interface="tf", diff_method="backprop")
def qnode(inputs, weights):
    for i in range(blocks):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.StronglyEntanglingLayers(weights[i], wires=range(n_qubits)) #STRONGLY ENTANGLING LAYERS
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weights_shape = (blocks, layers, n_qubits, 3) # Uncomment for Strongly entangling layers

# Make predictions 
y_pred = modelh.predict(X_test)

from matplotlib import pyplot
pyplot.plot(historyh.history['loss'], label='loss train')
pyplot.plot(historyh.history['val_loss'], label='loss val')
pyplot.legend()
pyplot.show()

#pyplot.plot(historyh.history['binary_accuracy'], label='accuracy train')
#pyplot.plot(historyh.history['val_binary_accuracy'], label='accuracy test')
#pyplot.plot(history.history['mse'], label='mse')
#pyplot.plot(history.history['mae'], label='mae')
#pyplot.legend()
#pyplot.show()

pyplot.plot(historyh.history['auc'], label='auc train')
pyplot.plot(historyh.history['val_auc'], label='auc val')
pyplot.legend()
pyplot.show()


y_pred = (modelh.predict(X_test) > 0.5).astype("int32")
from numpy import arange, meshgrid, hstack
plt.figure()
cm = plt.cm.RdBu_r
fig= plt.figure(figsize=(15,15))
# make data for decision regions
# define bounds of the domain
min1, max1 = X_test[:, 0].min()-0.2, X_test[:, 0].max()+0.2
min2, max2 = X_test[:, 1].min()-0.2, X_test[:, 1].max()+0.2
# define the x and y scale
x1grid = arange(min1, max1, 0.1)
x2grid = arange(min2, max2, 0.1)
# create all of the lines and rows of the grid
xx, yy = meshgrid(x1grid, x2grid)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# horizontal stack vectors to create x1,x2 input for the model
grid = hstack((r1,r2))
# make predictions for the grid
yhat = (modelh.predict(grid) > 0.5).astype("int32")
# reshape the predictions back into a grid
zz = yhat.reshape(xx.shape)


from numpy import arange, meshgrid, hstack
plt.figure()
fig= plt.figure(figsize=(1,1))
cm = plt.cm.RdBu_r
# plot decision regions
cnt = plt.contourf(xx, yy,zz, levels=np.arange(0, 1, 0.1), cmap=cm, alpha=0.8, extend="both")
plt.contour(xx, yy,zz, levels=[0.5], colors=("black",), linestyles=("--",), linewidths=(4.0,))


# plot data
plt.scatter(
    X_train[:, 0][y_train == 1],
    X_train[:, 1][y_train == 1],
    c="r",s=50,
    marker="o",
    edgecolors="k",
    label="class 1 Train",
)
plt.scatter(
       X_train[:, 0][y_train == 0],
       X_train[:, 1][y_train == 0],
    c="b",
    marker="o",s=50,
    edgecolors="k",
    label="class 0 train",
)

plt.title(label=LABELplot,
          fontsize=12,
          color="Black")
plt.xlabel('')
plt.ylabel('')
plt.legend(loc='upper left',fontsize=10)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
plt.show()


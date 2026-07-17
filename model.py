import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Reshape once (for CNN + augmentation)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
)
datagen.fit(x_train)

def mcDropoutPrediction(model, x, T=50):
    predictions = []

    for _ in range(T):
        pred = model(x, training=True)
        predictions.append(pred.numpy())

    predictions = np.array(predictions)

    mean_prediction = predictions.mean(axis=0)
    variance = predictions.var(axis=0)

    return mean_prediction, variance
    

# Model builder (fixed for LeakyReLU)
def buildModel(activation, optimizer):

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    if activation == "leaky_relu":
        model.add(layers.LeakyReLU(alpha=0.1))
    else:
        model.add(layers.Activation(activation))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    if activation == "leaky_relu":
        model.add(layers.LeakyReLU(alpha=0.1))
    else:
        model.add(layers.Activation(activation))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(128))
    if activation == "leaky_relu":
        model.add(layers.LeakyReLU(alpha=0.1))
    else:
        model.add(layers.Activation(activation))

    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# Compare function
def compare(cmpOpt, params, augmented):
    histories = {}
    mdls = {}

    for param in params: 
        print("Training with:", param)

        if cmpOpt:
            model = buildModel(activation='relu', optimizer=param)
        else:
            model = buildModel(activation=param, optimizer='adam')

        if(augmented):
            history = model.fit(
                datagen.flow(x_train, y_train, batch_size=32),
                epochs=10,
                validation_data=(x_test, y_test),
                steps_per_epoch=len(x_train)//32,
                verbose=1
            )
        else:
            history = model.fit(
                x_train, y_train,
                epochs=10,
                batch_size=32,
                validation_data=(x_test, y_test),
                verbose=1
            )

        histories[param] = history
        mdls[param] = model

    return histories, mdls

# =========================
# Compare Data Augmentation Impact
# Note: Activation and optimizer are fixed (adam + relu)
# Unaugmented
hist_no_aug, model_no_aug = compare(True, ['adam'], False)

# Augmented 
hist_aug, model_aug = compare(True, ['adam'], True)

# Accuracy Comparison
plt.figure()

plt.plot(hist_no_aug['adam'].history['val_accuracy'],
        linestyle='--',
        label='No Augmentation')

plt.plot(hist_aug['adam'].history['val_accuracy'],
        label='With Augmentation')

plt.title("Effect of Data Augmentation on Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.savefig("data_augmentation_validation_accuracy_comparison.png")
plt.close()

# Loss Comparison
plt.figure()

plt.plot(hist_no_aug['adam'].history['val_loss'],
        linestyle='--',
        label='No Augmentation')

plt.plot(hist_aug['adam'].history['val_loss'],
        label='With Augmentation')

plt.title("Effect of Data Augmentation on Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("data_augmentation_validation_loss_comparison.png")
plt.close()

# =========================
# Compare Optimizers (Unaugmented)
optimizers = ['adam', 'rmsprop', 'sgd']
opt_histories, opt_models = compare(True, optimizers, False)

plt.figure()
for opt in opt_histories:
    plt.plot(opt_histories[opt].history['val_accuracy'], label=opt)
plt.title("Optimizer Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("optimizer_accuracy_comparison.png")
plt.close()

plt.figure()
for opt in opt_histories:
    plt.plot(opt_histories[opt].history['val_loss'], label=opt)
plt.title("Optimizer Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("optimizer_loss_comparison.png")
plt.close()


# =========================
# Compare Activations (Unaugmented)
activations = ['relu', 'sigmoid', 'leaky_relu']
act_histories, act_models = compare(False, activations, False)

plt.figure()
for act in act_histories:
    plt.plot(act_histories[act].history['val_accuracy'], label=act)
plt.title("Activation Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("activation_accuracy_comparison.png")
plt.close()

plt.figure()
for act in act_histories:
    plt.plot(act_histories[act].history['val_loss'], label=act)
plt.title("Activation Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("activation_loss_comparison.png")
plt.close()

#=========================
#MC Dropout 

#train model 50 different time with random dropout units picked. 
mean_pred, variance = mcDropoutPrediction(
    model_no_aug['adam'],
    x_test,
    T=50
)

predictions = np.argmax(mean_pred, axis=1)
accuracy = np.mean(predictions == y_test)
print("MC Dropout Accuracy:", accuracy)

#create entropy graph
uncertainty = variance.mean(axis=1)
entropy = -np.sum(
    mean_pred * np.log(mean_pred + 1e-10),
    axis=1
)
plt.figure()
plt.hist(entropy, bins=40)
plt.xlabel("Predictive Entropy")
plt.ylabel("Count")
plt.title("MC Dropout Predictive Entropy")
plt.savefig("mc_dropout_entropy.png")
plt.close()

#create variance graph (for correct/incorrect prediction class)
correct = predictions == y_test

plt.figure()

plt.hist(
    uncertainty[correct],
    bins=40,
    alpha=0.6,
    label="Correct"
)

plt.hist(
    uncertainty[~correct],
    bins=40,
    alpha=0.6,
    label="Incorrect"
)

plt.xlabel("Predictive Variance")
plt.ylabel("Count")
plt.title("MC Dropout Uncertainty")
plt.legend()

plt.savefig("mc_dropout_uncertainty.png")
plt.close()
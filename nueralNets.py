import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt


def build_model(n_kernels=8, kernel_size=3, stride=2, n_dense=32):


    model = Sequential()
    model.add(Conv2D(n_kernels, (kernel_size, kernel_size), activation='relu', input_shape=(16, 16, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(stride, stride)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.0001))

    return model

def printError(trainErrors, testErrors, params, label):
    plt.plot(params, trainErrors, linestyle='-', linewidth=3.0, color='k',
             label='Training-Error')
    plt.plot(params, testErrors, linestyle='--', linewidth=3.0, color='y',
             label='Test-Error')
    plt.title('The effect of ' + label)
    plt.xlabel(label)
    plt.ylabel('Error')
    plt.legend(loc='upper right')
    plt.show()

def printParams(Params, variable, label):
    plt.plot(variable, Params, linestyle='-', linewidth=3.0, color='k')
    plt.title('The effect of ' + label)
    plt.xlabel(label)
    plt.ylabel('Params')
    plt.show()

if __name__ == '__main__':
    usps_data = pickle.load(open('usps.pickle', 'rb'))

    x_train = usps_data['x']['trn']
    y_train = usps_data['y']['trn']
    x_val = usps_data['x']['val']
    y_val = usps_data['y']['val']
    x_test = usps_data['x']['tst']
    y_test = usps_data['y']['tst']

    kernals = [1, 2, 4, 8, 16]
    kernel_size = [1, 2, 3, 4, 5]
    stride = [1, 2, 3, 4]
    n_dense = [16, 32, 64, 128]

    trainErrors = []
    testErrors = []
    Params = []

    #### kernals ####

    for i in range(len(kernals)):
        model = build_model(kernals[i], 3, 2, 32)
        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

        model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=2,
                  validation_data=(x_val, y_val), callbacks=[annealer])

        trainError = model.evaluate(x_train, y_train, batch_size=16)
        testError = model.evaluate(x_test, y_test, batch_size=16)

        trainErrors.append(trainError)
        testErrors.append(testError)
        Params.append(model.count_params())

    printError(trainErrors, testErrors, kernals, 'kernals')

    printParams(Params, kernals, 'kernals')

    #### kernel_size ####

    trainErrors = []
    testErrors = []
    Params = []

    for i in range(len(kernel_size)):
        model = build_model(8, kernel_size[i], 2, 32)

        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

        model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=2,
                  validation_data=(x_val, y_val), callbacks=[annealer])

        trainError = model.evaluate(x_train, y_train, batch_size=16)
        testError = model.evaluate(x_test, y_test, batch_size=16)

        trainErrors.append(trainError)
        testErrors.append(testError)
        Params.append(model.count_params())

    printError(trainErrors, testErrors, kernel_size, 'kernel_size')

    printParams(Params, kernel_size, 'kernel_size')

    #### stride ####

    trainErrors = []
    testErrors = []
    Params = []

    for i in range(len(stride)):
        model = build_model(8, 3, stride[i], 32)

        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

        model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=2,
                  validation_data=(x_val, y_val), callbacks=[annealer])

        trainError = model.evaluate(x_train, y_train, batch_size=16)
        testError = model.evaluate(x_test, y_test, batch_size=16)

        trainErrors.append(trainError)
        testErrors.append(testError)
        Params.append(model.count_params())

    printError(trainErrors, testErrors, stride, 'stride')

    printParams(Params, stride, 'stride')

    #### n_dense ####

    trainErrors = []
    testErrors = []
    Params = []

    for i in range(len(n_dense)):
        model = build_model(8, 3, 2, n_dense[i])

        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

        model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=2,
                  validation_data=(x_val, y_val), callbacks=[annealer])

        trainError = model.evaluate(x_train, y_train, batch_size=16)
        testError = model.evaluate(x_test, y_test, batch_size=16)

        trainErrors.append(trainError)
        testErrors.append(testError)
        Params.append(model.count_params())


    printError(trainErrors,testErrors,n_dense,'n_dense')

    printParams(Params, n_dense, 'n_dense')


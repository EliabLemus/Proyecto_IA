import math, csv
import numpy as np
import matplotlib.pyplot as chart

USAC_LAT=14.589246
USAC_LON=-90.551449
MUNICIPIOS = {}
 
class Data:
    def __init__(self, data_set_x, data_set_y, max_value=1):
        self.m = data_set_x.shape[1]
        self.n = data_set_x.shape[0]
        self.x = data_set_x / max_value  # escalacion de variables
        self.y = data_set_y

class NN_Model:
    def __init__(
        self, train_set, layers, alpha=0.3, iterations=300000, lambd=0, keep_prob=1
    ):
        self.data = train_set
        self.alpha = alpha
        self.max_iteration = iterations
        self.lambd = lambd
        self.kp = keep_prob
        # Se inicializan los pesos
        self.parametros = self.Inicializar(layers)

    def Inicializar(self, layers):
        parametros = {}
        L = len(layers)

        print("layers:", layers)
        for l in range(1, L):
            # np.random.randn(layers[l], layers[l-1])
            # Crea un arreglo que tiene layers[l] arreglos, donde cada uno de estos arreglos tiene layers[l-1] elementos con valores aleatorios
            # np.sqrt(layers[l-1] se saca la raiz cuadrada positiva de la capa anterior ---> layers[l-1]
            parametros["W" + str(l)] = np.random.randn(
                layers[l], layers[l - 1]
            ) / np.sqrt(layers[l - 1])
            parametros["b" + str(l)] = np.zeros((layers[l], 1))
            # print(layers[l], layers[l-1], np.random.randn(layers[l], layers[l-1]))
            # print(np.sqrt(layers[l-1]))
            # print(np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1]))

        return parametros

    def training(self, show_cost=False):
        self.bitacora = []
        for i in range(0, self.max_iteration):
            y_hat, order_temp = self.propagacion_adelante(self.data)
            cost = self.cost_function(y_hat)
            gradientes = self.propagacion_atras(order_temp)
            self.actualizar_parametros(gradientes)
            if i % 50 == 0:
                self.bitacora.append(cost)
                if show_cost:
                    print("Iteracion No.", i, "Costo:", cost, sep=" ")

    def propagacion_adelante(self, dataSet):
        # concatenar resultados:
        results = []
        results_dict = {}
        # Se extraen las entradas
        X = dataSet.x
        # cuento las capas existentes
        layers_count = int(len(self.parametros.keys()) / 2)
        A = 0
        for l in range(1, layers_count + 1):
            activation_name = "sigmoide" if l == layers_count else "tanh"
            dropout = False if l == layers_count else True
            WK = self.parametros["W" + str(l)]
            BK = self.parametros["b" + str(l)]
            if l == 1:
                Z = np.dot(WK, X) + BK
            else:
                Z = np.dot(WK, A) + BK
            A = self.activation_function(activation_name, Z)
            # Se aplica el Dropout Invertido

            if dropout:
                D = np.random.rand(
                    A.shape[0], A.shape[1]
                )  # Se generan número aleatorios para cada neurona
                D = (D < self.kp).astype(
                    int
                )  # Mientras más alto es kp mayor la probabilidad de que la neurona permanezca
                A *= D
                A /= self.kp
                results.append(Z)
                results.append(A)
                results.append(D)
                results_dict["Z" + str(l)] = Z
                results_dict["A" + str(l)] = A
                results_dict["D" + str(l)] = D
            else:
                results.append(Z)
                results.append(A)
                results_dict["Z" + str(l)] = Z
                results_dict["A" + str(l)] = A

        Aresult = results[len(results) - 1]
        T1 = tuple(results)

        return Aresult, results_dict

    def propagacion_atras(self, order_temp):
        gradientes_result = {}
        # Se obtienen los datos
        m = self.data.m
        Y = self.data.y
        X = self.data.x

        layers_count = int(len(self.parametros.keys()) / 2)
        dZ = 0

        for l in range(layers_count, 0, -1):
            Wk = self.parametros["W" + str(l)]

            Wknext = self.parametros["W" + str(l + 1)] if l < layers_count else 0
            Ak = order_temp["A" + str(l)]
            Dk = order_temp["D" + str(l)] if l < layers_count else 0
            Akprev = order_temp["A" + str(l - 1)] if l > 1 else 0

            if l == layers_count:
                # dZ3 = A3 - Y
                # dW3 = (1 / m) * np.dot(dZ3, A2.T) + (self.lambd / m) * W3
                # db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
                ## Tercera capa
                dZ = Ak - Y
                dW = (1 / m) * np.dot(dZ, Akprev.T) + (self.lambd / m) * Wk
                db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
                gradientes_result["dZ" + str(l)] = dZ
                gradientes_result["dW" + str(l)] = dW
                gradientes_result["db" + str(l)] = db
            elif l == 1:
                # Derivadas parciales de la primera capa
                # dA1 = np.dot(W2.T, dZ2)
                # dA1 *= D1
                # dA1 /= self.kp
                # dZ1 = np.multiply(dA1, np.int64(A1 > 0))
                # dW1 = 1./m * np.dot(dZ1, X.T) + (self.lambd / m) * W1
                # db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

                dA = np.dot(Wknext.T, dZ)
                dA *= Dk
                dA /= self.kp
                dZ = np.multiply(dA, np.int64(Ak > 0))
                dW = 1.0 / m * np.dot(dZ, X.T) + (self.lambd / m) * Wk
                db = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
                gradientes_result["dA" + str(l)] = dA
                gradientes_result["dZ" + str(l)] = dZ
                gradientes_result["dW" + str(l)] = dW
                gradientes_result["db" + str(l)] = db
            else:
                # Derivadas parciales de la segunda capa
                # dA2 = np.dot(W3.T, dZ3)
                # dA2 *= D2
                # dA2 /= self.kp
                # dZ2 = np.multiply(dA2, np.int64(A2 > 0))
                # dW2 = 1. / m * np.dot(dZ2, A1.T) + (self.lambd / m) * W2
                # db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

                dA = np.dot(Wknext.T, dZ)
                dA *= Dk
                dA /= self.kp
                dZ = np.multiply(dA, np.int64(Ak > 0))
                dW = 1.0 / m * np.dot(dZ, Akprev.T) + (self.lambd / m) * Wk
                db = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
                gradientes_result["dA" + str(l)] = dA
                gradientes_result["dZ" + str(l)] = dZ
                gradientes_result["dW" + str(l)] = dW
                gradientes_result["db" + str(l)] = db

        return gradientes_result

    def actualizar_parametros(self, grad):
        # Se obtiene la cantidad de pesos
        L = len(self.parametros) // 2
        for k in range(L):
            self.parametros["W" + str(k + 1)] -= self.alpha * grad["dW" + str(k + 1)]
            self.parametros["b" + str(k + 1)] -= self.alpha * grad["db" + str(k + 1)]

    def cost_function(self, y_hat):
        # Se obtienen los datos
        Y = self.data.y
        m = self.data.m
        # Se hacen los calculos
        temp = np.multiply(-np.log(y_hat), Y) + np.multiply(-np.log(1 - y_hat), 1 - Y)
        result = (1 / m) * np.nansum(temp)
        # Se agrega la regularizacion L2
        if self.lambd > 0:
            L = len(self.parametros) // 2
            suma = 0
            for i in range(L):
                suma += np.sum(np.square(self.parametros["W" + str(i + 1)]))
            result += (self.lambd / (2 * m)) * suma
        return result

    def predict(self, dataSet):
        # Se obtienen los datos
        m = dataSet.m
        Y = dataSet.y

        p = np.zeros((1, m), dtype=np.int)
        # Propagacion hacia adelante
        y_hat, temp = self.propagacion_adelante(dataSet)
        # Convertir probabilidad
        for i in range(0, m):
            p[0, i] = 1 if y_hat[0, i] > 0.5 else 0
        exactitud = np.mean(
            (
                p[0, :]
                == Y[
                    0,
                ]
            )
        )
        print("Exactitud: " + str(exactitud))
        return exactitud

    def activation_function(self, name, x):
        result = 0
        if name == "sigmoide":
            result = 1 / (1 + np.exp(-x))
        elif name == "tanh":
            result = np.tanh(x)
        elif name == "relu":
            result = np.maximum(0, x)

        # print('name:', name, 'result:', result)
        return result


class Row:
    def __init__(self,gender,age,enrollmentYear,distanceFromUniversity,state):
        self.gender = gender
        self.age = age
        self.enrollmentYear = enrollmentYear
        self.distanceFromUniversity = distanceFromUniversity
        self.state = state

def getHaversineDistances(lat1=0, lon1=0, lat2=0, lon2=0):
    """
    lat1: latitud punto 1
    lon1: longitud punto 1
    lat2: latitud punto 2
    lon2: longitud punto 2
    """
    R = 6372.795477598
    rad = math.pi / 180
    DeltaLat = float(lat1 - lat2)
    DeltaLon = float(lon1 - lon2)
    a = (math.sin(rad * DeltaLat / 2)) ** 2 + math.cos(rad * lat1) * math.cos(
        rad * lat2
    ) * (math.sin(rad * DeltaLon / 2)) ** 2
    distance = 2 * R * math.asin(math.sqrt(a))
    return distance 


def plot_field_data(data_x, data_y):
    chart.scatter(data_x[0, :], data_y[1, :], c=data_y, s=25, cmap=chart.cm.Spectral)
    chart.show()


def show_Model(models):
    for model in models:
        chart.plot(model.bitacora, label=str(model.alpha))
    chart.ylabel("Costo")
    chart.xlabel("Iteraciones")
    legend = chart.legend(loc="upper center", shadow=True)
    chart.show()


def getData():
    result = []
    # limit = 500
    # random limits = -50,50

    limit = 500
    for k in range(0, limit):
        # k = x
        x = np.random.randint(-50, 50)
        result.append([x, getFunctionValue(x)])
    return np.array(result)


def getFunctionValue(x):
    # x^2 - sin(x) + 1
    return ((x ** 2) / 100) - math.sin(x) + 1


def example():
    ONLY_SHOW = False
    data_set = getData()
    print(data_set.shape)
    # divido 70/30
    slice_point = int(data_set.shape[0] * 0.7)
    # # # print('slice_point:', slice_point)
    train_set_x = data_set[0:slice_point]
    train_set_y = np.random.randint(2, size=train_set_x.shape[0])
    
    test_set_x = data_set[slice_point:]
    test_set_y = np.random.randint(2, size=test_set_x.shape[0])

    train_set_x = train_set_x.T
    train_set_y = train_set_y.T
    test_set_x = test_set_x.T
    test_set_y = test_set_y.T

    if ONLY_SHOW:
        plot_field_data(train_set_x, train_set_y)
        # Plotter.plot_field_data(val_X, val_Y)
        print("Entradas de entrenamiento:", train_set_x.shape, sep=" ")
        print("Salidas de entrenamiento:", train_set_y.shape, sep=" ")
        print("Entradas de validacion:", test_set_x.shape, sep=" ")
        print("Salidas de validacion:", test_set_y.shape, sep=" ")
        exit()

    train = Data(train_set_x, train_set_y, 100)
    test = Data(test_set_x, test_set_y, 100)
    layers = [train.n, 10, 5, 1]

    # Se define el modelo
    Model1 = NN_Model(
        train, layers, alpha=0.01, iterations=50000, lambd=0.7, keep_prob=1
    )
    Model1.training(True)
    show_Model([Model1])
    print("Entrenamiento Modelo 1")
    Model1.predict(train)
    print("Validacion Modelo 1")
    Model1.predict(test)

    plot_field_data(train_set_x, train_set_y)
    print("Entradas de entrenamiento:", train_set_x.shape, sep=" ")
    print("Salidas de entrenamiento:", train_set_y.shape, sep=" ")
    print("Entradas de validacion:", test_set_x.shape, sep=" ")
    print("Salidas de validacion:", test_set_y.shape, sep=" ")

def getCoordinatesMunicipio(cod_depto=0,cod_municipio=0):
    lat = float(MUNICIPIOS.get(cod_depto).get(cod_municipio).get('Lat'))
    lon = float(MUNICIPIOS.get(cod_depto).get(cod_municipio).get('Lon'))
    return (lat,lon)
def getDistanceFromUniversity(cod_depto=0,cod_municipio=0):
    mun_coordinates = getCoordinatesMunicipio(cod_depto=cod_depto, cod_municipio=cod_municipio)
    return getHaversineDistances(lat1=mun_coordinates[0],lon1=mun_coordinates[1],lat2=USAC_LAT,lon2=USAC_LON)
def getMunicipiosDict():
    with open('Datasets/Municipios.csv', 'rt') as f:
        reader = csv.DictReader(f, delimiter = ',')
        data =  list(reader)
        byDeptos = {}
        byMunic = {}
        for i in range(1,23):
            municipios = [a for a in data if a['Depto'] == str(i) ]
            for a in municipios:
                byMunic[int(a.get('Muni'))] = a
            byDeptos[i] = byMunic
    
    return byDeptos
def getArray(gender,age,enrollmentYear,distanceFromUniversity,state):
    result = []
    #Arreglo:
    #[male,female,age,year,distance,traslado,activo]
    if gender == 'MASCULINO':
        result.append(1)
        result.append(0)
    elif gender == 'FEMENINO':
        result.append(0)
        result.append(1)
    else:
        result.append(0)
        result.append(0)
    
    result.append(float(age)) #Escalar 
    result.append(float(enrollmentYear)) #escalar
    result.append(float(distanceFromUniversity)) #escalar
    if state == 'Traslado':
        result.append(0)
    elif state == 'Activo':
        result.append(1)
    else:
        result.append(0)
     
     
    return result   
    
             
def getDataset():
    list_dataset = []
    with open('Datasets/Dataset.csv', "rt", encoding='iso-8859-1') as f:
        reader = csv.DictReader(f, delimiter = ',')
        for k in reader:
            distance = getDistanceFromUniversity(cod_depto=int(k['cod_depto']), cod_municipio=int(k['cod_muni']))
            list_dataset.append(getArray(gender=k['Genero'],age=k['edad'],enrollmentYear=k['Año'],distanceFromUniversity=distance,state=k['Estado']))
        return np.array([escalateVariables(row=k) for k in list_dataset])


def escalateVariables(row=[]):
    target_positions = [2,3,4]
    #[male,female,age,year,distance,traslado/activo]
    #2 age: 
    max_value = float(max(row))
    min_value = float(min(row))
    for k in target_positions:
        to_escalate = row[k]
        new_value = (to_escalate - min_value)/(max_value-min_value)
        row[k]=new_value
    return row
    
if __name__ == "__main__":
    MUNICIPIOS = getMunicipiosDict()

    # result = getHaversineDistances(
    #     6.27823496943, -75.5694735416, 6.28331696378, -75.5689742567
    # )
    # print("resultado:", result)
    data_set = getDataset()
    print(data_set.shape)
    # divido 70/30
    slice_point = int(data_set.shape[0] * 0.7)
    print('slice_point:', slice_point)
    #[male,female,age,year,distance,traslado,activo]
    
    train_set_x = data_set[0:slice_point, :5]
    train_set_y = data_set[0:slice_point, 5:]
    
    # train_set_y = np.random.randint(2, size=train_set_x.shape[0])
    test_set_x = data_set[slice_point:, :5]
    test_set_y = data_set[slice_point:, 5:]

    

    train_set_x = train_set_x.T
    train_set_y = train_set_y.T
    test_set_x = test_set_x.T
    test_set_y = test_set_y.T
    
    print('train_set_x: ',train_set_x.shape)
    print('train_set_y: ',train_set_y.shape)
    print('test_set_x:', test_set_x.shape)
    print('test_set_y: ', test_set_y.shape)
    # plot_field_data(train_set_x, train_set_y)
    
    train = Data(train_set_x, train_set_y)
    test = Data(test_set_x, test_set_y)
    layers = [train.n, 7, 7,5, 1]

    # Se define el modelo
    Model1 = NN_Model(
        train, layers, alpha=0.005, iterations=10000, lambd=0.007, keep_prob=1
    )
    Model1.training(True)
    show_Model([Model1])
    print("Entrenamiento Modelo 1")
    Model1.predict(train)
    print("Validacion Modelo 1")
    Model1.predict(test)
    

    

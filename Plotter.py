import matplotlib.pyplot as chart
chart.switch_backend('Agg')


def plot_field_data(data_x, data_y):
    chart.scatter(data_x[0, :], data_x[1, :], c=data_y, s=40, cmap=chart.cm.Spectral)
    chart.show()

def show_Model(models):
    for model in models:
        chart.plot(model.bitacora, label=str(model.alpha))
    chart.ylabel('Costo')
    chart.xlabel('Iteraciones')
    legend = chart.legend(loc='upper center', shadow=True)
    chart.savefig('static/ModelGraphs/best_model.png')
    
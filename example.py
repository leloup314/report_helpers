import numpy as np
from report_helpers import plot


def example_plot():
    """
    Makes four example plots with fits
    """

    # Make four example fit funcs
    def func1(x, a, b):
        return a * np.power(x, b)

    def func2(x, a, b):
        return a * np.sqrt(x) + b

    def func3(x, a, b):
        return a * np.exp(b * x)
        
    def func4(I, R, U_0):
        return R * I + U_0

    # Init a,b for data
    a = 7.5845
    b = 1.1298e-1

    # Make noise as upper/lower limits as random percentage of input
    noise_factor = 0.1  # up to 10% noise

    # Make four datasets with gaussian noise
    x = [i for i in range(1, 100)]
    x_err = [i * 0.1 for i in x]
    y1 = [(a * i ** b) + (a * i ** b) * np.random.uniform(-noise_factor, noise_factor) for i in range(1, 100)]
    y2 = [(a * i ** 0.5 + b) + (a * i ** 0.5 + b) * np.random.uniform(-noise_factor, noise_factor) for i in range(1, 100)]
    y3 = [(a * np.exp(b * i)) + (a * np.exp(b * i)) * np.random.uniform(-noise_factor, noise_factor) for i in range(1, 100)]
    y4 = [(a * i + b) + (a * i + b) * np.random.uniform(-noise_factor, noise_factor) for i in range(1, 100)]
    y1_err = [i * noise_factor for i in y1]
    y2_err = [i * noise_factor for i in y2]
    y3_err = [i * noise_factor for i in y3]
    y4_err = [i * noise_factor for i in y4]

    data1 = {'data': [x, y1, x_err, y1_err], 'label': 'Example data 1'}
    fit1 = {'data': [x, y1, y1_err], 'func': func1, 'label': '$f(x)=a\cdot x^{b}$', 'start_parameters': (1, 1e-4),
            'precision': 1}

    data2 = {'data': [x, y2, x_err, y2_err], 'label': 'Example data 2'}
    fit2 = {'data': [x, y2, y2_err], 'func': func2, 'label': '$f(x)=a\cdot\sqrt{x} + b$', 'start_parameters': (1, 1e-4),
            'precision': 2}

    data3 = {'data': [x, y3, x_err, y3_err], 'label': 'Exponential Growth'}
    fit3 = {'data': [x, y3, y3_err], 'func': func3, 'label': '$f(x)=a\cdot e^{xb}$', 'start_parameters': (1, 1e-4),
            'precision': 3}
        
    data4 = {'data': [x, y4, x_err, y4_err], 'label': 'Ohms Law'}
    fit4 = {'data': [x, y4, y4_err], 'units': [r'$\frac{mV}{\mu A}$', 'mV'], 'func': func4, 'label': r'$U(I)=R \cdot I$', 'start_parameters': (1, 1e-4),
            'precision': 3}

    data5 = {'data': [[x, y1, x_err, y1_err], [x, y2, x_err, y2_err]], 'label': ['Example data 1', 'Example data 2']}
    fit5 = {'data': [[x, y1, y1_err], [x, y2, y2_err]], 'func': [func1, func2],
            'label': ['$f(x)=a\cdot x^{b}$', '$f(x)=a\cdot\sqrt{x} + b$'], 'start_parameters': [(1, 1e-4), (1, 1e-4)],
            'precision': [3, 4]}

    # Make plots
    plot(data=data1, fit=fit1, output_pdf='./example_plot_1.pdf')
    plot(data=data2, fit=fit2, output_pdf='./example_plot_2.pdf')
    plot(data=data3, fit=fit3, output_pdf='./example_plot_3.pdf')
    plot(title='Ohms Law with units', x_label=r'$I\ /\ \mu A$', y_label='U / mV', data=data4, fit=fit4, output_pdf='./example_plot_4.pdf')
    plot(title='Multiple plots with logscale', data=data5, y_scale='log', legend_outside=True, fit=fit5, output_pdf='./example_plot_5.pdf')
        
    
if __name__ == '__main__':
    example_plot()
    

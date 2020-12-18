import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

save = False

if save:
    pdf = matplotlib.backends.backend_pdf.PdfPages("experiment1.pdf")


def show(plot, title=None):
    plt.subplots_adjust(top=0.9)
    if title:
        plot.fig.suptitle(title, size=20)
    if save:
        pdf.savefig()
    else:
        plt.show()
    plt.close()


def close_pdf():
    if save:
        pdf.close()

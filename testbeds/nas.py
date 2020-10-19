from nasbench import api
import numpy as np
from testbeds.main_usecase import Testbed


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

class NAS(Testbed):

    def __init__(self, path):
        super(NAS, self).__init__()

        PATH = path
        self.OPS = [INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT]

        self.nasbench = api.NASBench(PATH)
        self.size= 7
        self.var_er = None
        self.D=21

    def simulate(self, w_orig, eval=False):
        if eval:
            matrix = self.transform(w_orig)
            cell = api.ModelSpec(
                matrix=matrix,
                ops=self.OPS
            )

            if self.nasbench.is_valid(cell):
                return cell
            else:
                print('invalid cell')
                return 1

        else:
            matrix = self.transform(w_orig)
            # while True:
            cell = api.ModelSpec(
                matrix=matrix,
                ops = self.OPS
            )

            if self.nasbench.is_valid(cell):
                return cell

            else:
                return 1

    def transform(self, w_orig):
        w = w_orig.copy()

        # convert w_orig to matrix form
        W_m = []
        a = 0
        for x in range(self.size - 1, 0, -1):
            W_m.append(w[a:(a + x)])
            a = (a + x)

        # create a search matrix in line with the work
        matrix = np.zeros((self.size, self.size))
        for enu, row in enumerate(W_m):
            matrix[enu, enu + 1:] += row

        matrix = matrix.astype(int)

        return matrix


    def extract(self, matrix):
        w_cor = []
        for idx, row in enumerate(matrix):
            if idx < matrix.shape[0] - 1:
                w_cor.append(row[idx + 1:])
        return np.hstack(w_cor)

    def distance(self, input, eval=False):
        # convert to a minimization problem
        if input==1:
            return 1
        else:
            if eval:
                return 1 - self.nasbench.query(input)['test_accuracy']
            else:
                return 1 - self.nasbench.query(input)['validation_accuracy']

    def prior(self, theta):
        return np.exp(-np.mean(theta))
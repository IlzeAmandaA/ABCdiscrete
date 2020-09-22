from nasbench import api
import numpy as np
from testbeds.main_usecase import Testbed

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

class NAS(Testbed):

    def __init__(self):
        super(NAS, self).__init__()

        PATH = '/home/ilze/MasterThesis/nas/nasbench_only108.tfrecord'
        self.OPS = [INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT]

        self.nasbench = api.NASBench(PATH)
        self.size= 7
        self.var_er = None
        self.D=21

    def simulate(self, w_orig, eval=False):
        #check if requrieent of max 9 edges is met
        if np.sum(w_orig)>9:
            pass

        #convert w_orig to matrix form
        W_m=[]
        a=0
        for x in range(self.size -1, 0, -1):
            W_m.append(w_orig[a:(a+x)])
            a = (a+x)

        #create a search matrix in line with the work
        matrix = np.zeros((self.size,self.size))
        for enu, row in enumerate(W_m):
            matrix[enu+1:] += row

        if eval:
            pass
        else:
            cell = api.ModelSpec(
                matrix = matrix,
                ops = self.OPS
            )

            data  = self.nasbench.query(cell)

        return data['validation_accuracy']


    def distance(self, input):
        #convert to a minimization problem
        return 1 - input

    def prior(self, theta):
        return np.exp(-np.mean(theta))
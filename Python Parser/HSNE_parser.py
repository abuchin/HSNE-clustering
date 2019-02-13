# Functions for reading HSNE hierarchy
import struct
import time
from scipy.sparse import coo_matrix
from HSNE import HSNE, DataScale, SubScale



def read_uint_vector(handle):
    '''
    Read unsigned int vector from HDI binary file
    :param handle: _io.BufferedReader (object result from calling native Python open() )
    :return: list
    '''
    vectorlength = struct.unpack('i', handle.read(4))[0]
    vector = list(struct.unpack('i' * vectorlength, handle.read(4 * vectorlength)))
    return vector


def read_scalar_vector(handle):
    '''
    Read float vector from HDI binary file
    :param handle: _io.BufferedReader (object result from calling native Python open() )
    :return: list
    '''
    vectorlength = struct.unpack('i', handle.read(4))[0]
    vector = list(struct.unpack('i' * vectorlength, handle.read(4 * vectorlength)))
    return vector


def read_HSNE_binary(filename, verbose=True):
    '''
    Read HSNE binary from file and construct HSNE object with top- and subscales
    :param filename: str, file to read
    :param verbose: bool, controls verbosity of parser
    :return: HSNE object
    '''
    logger = Logger(verbose)
    longtic = time.time()
    with open(filename, 'rb') as handle:
        majorversion = struct.unpack('f', handle.read(4))[0]  # Never used
        minorversion = struct.unpack('f', handle.read(4))[0]  # Never used
        numscales = int(struct.unpack('f', handle.read(4))[0])
        scalesize = int(struct.unpack('f', handle.read(4))[0])
        logger.log("Number of scales %i" % numscales)
        hierarchy = HSNE(numscales)
        logger.log("Start reading first scale of size %i" % scalesize)
        tmatrix = read_sparse_matrix(handle)
        logger.log("Done reading first scale..")
        hierarchy[0] = DataScale(num_scales=numscales, tmatrix=tmatrix)
        for i in range(1, numscales):
            hierarchy[i] = build_subscale(handle, i, numscales, logger)
        print('Total time spent parsing hierarchy and building objects: %f' % (time.time() - longtic))
        return hierarchy


def read_sparse_matrix(handle):
    cols = []
    rows = []
    weights = []
    numrows = struct.unpack('i' , handle.read(4))[0]
    shape = numrows
    for rownum in range(numrows):
        rowlen = struct.unpack('i', handle.read(4))[0]
        row = list(struct.unpack("if" * rowlen, handle.read(8 * rowlen)))
        cols += row[::2]
        weights += row[1::2]
        rows += [rownum] * rowlen
    return coo_matrix((weights, (rows, cols)), shape=(shape, shape))


def build_subscale(handle, i, numscales, logger):
    '''
    :param handle: _io.BufferedReader (object result from calling native Python open() )
    :param i: int, current scale
    :param numscales: total number of scales
    :param logger: Logger object
    :return: Subscale
    '''
    logger.log("\nNext scale: %i" % i)
    scalesize = int(struct.unpack('f', handle.read(4))[0])
    logger.log("Scale size: %i" % scalesize)
    logger.log("Reading transmatrix..")
    tmatrix = read_sparse_matrix(handle)
    logger.log("Reading landmarks of scale to original data..")
    lm_to_original = read_uint_vector(handle)
    logger.log("Reading landmarks to previous scale..")
    lm_to_previous = read_uint_vector(handle)
    logger.log("Reading landmark weights..")
    lm_weights = read_scalar_vector(handle)
    logger.log("Reading previous scale to current scale..")
    previous_to_current = read_uint_vector(handle)
    logger.log("Reading area of influence..")

    area_of_influence = read_sparse_matrix(handle)

    subscale = SubScale(scalenum=i,
                        num_scales=numscales,
                        tmatrix=tmatrix,
                        lm_to_original=lm_to_original,
                        lm_to_previous=lm_to_previous,
                        lm_weights=lm_weights,
                        previous_to_current=previous_to_current,
                        area_of_influence=area_of_influence
                        )
    return subscale


class Logger(object):
    # Message printer that can be turned off by initializing it with enabled=False
    def __init__(self, enabled):
        self._enabled = enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def log(self, message):
        if self._enabled:
            print(message)

if __name__ == "__main__":
    import sys
    read_HSNE_binary(sys.argv[1], verbose=False)

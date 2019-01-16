# Functions for reading HSNE hierarchy
import struct
import time
from scipy.sparse import coo_matrix
from HSNE import HSNE, DataScale, SubScale


def read_trans_matrix(handle):
    '''
    Read transition matrix from HDI binary file
    :param handle: _io.BufferedReader (object result from calling native Python open() )
    :return: list
    '''
    matrix1d = []
    rows = struct.unpack('i', handle.read(4))[0]
    matrix1d.append(rows)
    for repeat in range(rows):
        rowlen = struct.unpack('i', handle.read(4))[0]
        matrix1d.append(rowlen)
        for fields in range(rowlen):
            matrix1d.append(struct.unpack('i', handle.read(4))[0])
            matrix1d.append(struct.unpack('f', handle.read(4))[0])
    return matrix1d


def read_uint_vector(handle):
    '''
    Read unsigned int vector from HDI binary file
    :param handle: _io.BufferedReader (object result from calling native Python open() )
    :return: list
    '''
    vectorlength = struct.unpack('i', handle.read(4))[0]
    vector = []
    for i in range(vectorlength):
        vector.append(struct.unpack('i', handle.read(4))[0])
    return vector


def read_scalar_vector(handle):
    '''
    Read float vector from HDI binary file
    :param handle: _io.BufferedReader (object result from calling native Python open() )
    :return: list
    '''
    vectorlength = struct.unpack('i', handle.read(4))[0]
    vector = []
    for i in range(vectorlength):
        vector.append(struct.unpack('f', handle.read(4))[0])
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
        tmatrix = read_trans_matrix(handle)
        tmatrix = hdi_to_sparse(tmatrix)
        logger.log("Done reading first scale..")
        hierarchy[0] = DataScale(num_scales=numscales, tmatrix=tmatrix)
        for i in range(1, numscales):
            hierarchy[i] = build_subscale(handle, i, numscales, logger)
        logger.log('Total time spent parsing hierarchy and building objects: %f' % (time.time() - longtic))
        return hierarchy


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
    tmatrix = hdi_to_sparse(read_trans_matrix(handle))
    logger.log("Reading landmarks of scale to original data..")
    lm_to_original = read_uint_vector(handle)
    logger.log("Reading landmarks to previous scale..")
    lm_to_previous = read_uint_vector(handle)
    logger.log("Reading landmark weights..")
    lm_weights = read_scalar_vector(handle)
    logger.log("Reading previous scale to current scale..")
    previous_to_current = read_uint_vector(handle)
    logger.log("Reading area of influence..")
    tt = time.time()

    area_of_influence = hdi_to_sparse(read_trans_matrix(handle))
    logger.log('Time spent converting 1D to sparse: %f' % (time.time() - tt))

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


def hdi_to_sparse(hdidata):
    '''
    Convert 1D array to sparse matrix, output from read_trans_matrix() desired
    :param hdidata: list
    :return: scipy.sparse coo_matrix
    '''
    hdidata = hdidata[::-1]
    columns = []
    rows = []
    edgeweights = []
    shape = int(hdidata.pop())
    for rownum in range(shape):
        for field in range(int(hdidata.pop())):
            rows.append(rownum)
            columns.append(hdidata.pop())
            edgeweights.append(hdidata.pop())
    return coo_matrix((edgeweights, (rows, columns)), shape=(shape, shape))


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

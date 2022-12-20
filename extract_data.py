import tarfile
import os

data = tarfile.open(os.getcwd() + '/Data/dsgdb9nsd.xyz.tar.bz2', mode = 'r:bz2')

data.extractall(os.getcwd() + '/Data/xyz_wrong_format')

data.close()
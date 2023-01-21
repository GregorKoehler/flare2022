import numpy as np
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import subfiles
from nnunet.configuration import default_num_threads


def change_dtype(npy_file: str, target_dtype=np.uint8) -> None:
    np.save(npy_file, np.load(npy_file).astype(target_dtype))


def load_and_change_all(folder: str) -> None:
    p = Pool(default_num_threads)
    npy_files = subfiles(folder, suffix='.npy')
    res = p.map_async(change_dtype, npy_files)
    p.close()
    p.join()


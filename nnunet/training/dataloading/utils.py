import os
import subprocess


def get_allowed_n_proc_DA():
    hostname = subprocess.getoutput(["hostname"])
    if "nnUNet_n_proc_DA" in os.environ.keys():
        return int(os.environ["nnUNet_n_proc_DA"])
    if hostname in ["hdf19-gpu16", "hdf19-gpu17", "e230-AMDworkstation"]:
        return 16, hostname
    if hostname in ["E230-PC04", "e132-pc07"]:
        return 4, hostname
    if hostname in ["workstation", "e230-pc26"]:
        return 12, hostname
    if hostname.startswith("hdf19-gpu") or hostname.startswith("e071-gpu"):
        return 12, hostname
    elif hostname.startswith("e230-dgx1"):
        return 10, hostname
    elif hostname.startswith("hdf18-gpu") or hostname.startswith("e132-comp"):
        return 16, hostname
    elif hostname.startswith("e230-dgx2"):
        return 6, hostname
    elif hostname.startswith("e230-dgxa100-") or hostname.startswith("lsf22-gpu"):
        return 16, hostname
    else:
        return None, hostname

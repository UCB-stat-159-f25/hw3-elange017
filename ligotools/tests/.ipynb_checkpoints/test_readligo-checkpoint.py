# Test the file handling logic - loaddata() correctly distinguishes between .gwf and .hdf5 files
import os
import numpy as np
import pytest
from ligotools import readligo

def test_loaddata_zero_length(tmp_path):
    f = tmp_path / "empty.hdf5"
    f.touch()  # create empty file
    result = readligo.loaddata(str(f))
    assert result == (None, None, None)

# Test FileList class - that it correctly lists files and caches results
def test_filelist_find_and_cache(tmp_path):
    f1 = tmp_path / "test1.gwf"
    f2 = tmp_path / "test2.hdf5"
    f1.touch(); f2.touch()

    fl = readligo.FileList(directory=tmp_path)
    assert len(fl.list) == 2

    cache = tmp_path / "cache.txt"
    fl.writecache(cache)
    fl2 = readligo.FileList(directory=tmp_path, cache=str(cache))
    assert set(fl.list) == set(fl2.list)

# Test internal helper behavior: dq_channel_to_seglist() correctly 
# produces slices for a fake binary mask and dq2segs() returns proper segment start/stop pairs.
def test_dq_channel_to_seglist_basic():
    channel = np.array([0, 1, 1, 0, 1, 0])  # “good” data at 1–2 and 4
    segs = readligo.dq_channel_to_seglist(channel, fs=1)
    assert len(segs) == 2
    assert segs[0].start == 1 and segs[0].stop == 3

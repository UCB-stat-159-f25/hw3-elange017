import numpy as np
import os
import pytest
from ligotools.utils import whiten, write_wavfile, reqshift  # import your functions

def test_whiten_basic():
    """Test that whiten returns an array of the same length and type as input."""
    dt = 1/4096
    strain = np.random.randn(4096)  # 1 second of fake data
    psd = lambda f: np.ones_like(f)  # flat PSD
    
    white = whiten(strain, psd, dt)
    
    # Checks
    assert isinstance(white, np.ndarray)
    assert len(white) == len(strain)
    assert not np.all(white == 0)  # should not be all zeros

def test_write_wavfile_creates_file(tmp_path):
    """Test that write_wavfile creates a file with correct dtype."""
    filename = tmp_path / "test.wav"
    fs = 4096
    data = np.random.randn(1024)
    
    write_wavfile(str(filename), fs, data)
    
    # Check that file was created
    assert os.path.exists(filename)
    
    # Optionally, read back and check dtype
    from scipy.io import wavfile
    rate, wav_data = wavfile.read(filename)
    assert rate == fs
    assert wav_data.dtype == np.int16

# Test reqshift if you have it
def test_reqshift_basic():
    """Test that reqshift returns an array of the same shape."""
    dt = 1/4096
    strain = np.random.randn(1024)
    fshift = 10  # Hz
    
    shifted = reqshift(strain, fshift, dt)
    
    assert isinstance(shifted, np.ndarray)
    assert shifted.shape == strain.shape

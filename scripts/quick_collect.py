from pathlib import Path
import tifffile
import numpy as np

a = list(Path.cwd().glob('*.tif'))
out = []
for f in a:
    out.append(tifffile.imread(f))

outarr = np.array(out)
tifffile.imwrite('out.tif', outarr)
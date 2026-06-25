import h5py, time, sys
SRC = "/home/furka/compbio/fast5/hsapiens_single/PAO89685_pass__2264ba8c_afee3a87_1.0_275.fast5"
DST = "/home/furka/compbio/fast5/hsapiens_subset800/subset_800.fast5"
N = 800
t = time.time()
src = h5py.File(SRC, "r")
dst = h5py.File(DST, "w")
for k, v in src.attrs.items():
    dst.attrs[k] = v
groups = list(src.keys())
reads = [k for k in groups if k.startswith("read_")]
nonreads = [k for k in groups if not k.startswith("read_")]
for k in nonreads:
    src.copy(k, dst)
for k in reads[:N]:
    src.copy(k, dst)
dst.close()
src.close()
v = h5py.File(DST, "r")
nread = sum(1 for k in v if k.startswith("read_"))
v.close()
print("copied %d reads + %d nonread items in %.1fs; subset has %d read groups"
      % (min(N, len(reads)), len(nonreads), time.time() - t, nread))

import os
from os import listdir

for f in listdir("csvs_a_pezzi_unix/"):
    temp_file = f"csvs_a_pezzi_unix/{f}.tmp"
    os.system(f"LC_CTYPE=C sed 's/\\r//' csvs_a_pezzi_unix/{f} > {temp_file}")
    os.rename(temp_file, f"csvs_a_pezzi_unix/{f}")

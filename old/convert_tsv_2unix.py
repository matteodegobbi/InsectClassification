from os import listdir
import os
for f in listdir("csvs_a_pezzi_unix/"):
  os.system(f"sed 's/\r//' csvs_a_pezzi_unix/{f} | sponge csvs_a_pezzi_unix/{f}" )
  

import urllib.request 
import os.path
import os

input_options = open("input.txt", "r")
request_string = ''

for x in input_options:
  request_string = request_string + '&' + x.rstrip('\n')

url = f"http://v3.boldsystems.org/index.php/API_Public/specimen?{request_string}&format=tsv"

fname = 'raw_data.tsv'

if not os.path.isfile(fname):
    try:
        urllib.request.urlretrieve(url, fname)
    except Exception as e:
        print(f"Error: {e}")


temp_file = 'raw_data_unix.tmp'
os.system(f"LC_CTYPE=C sed 's/\\r//' 'raw_data.tsv' > {temp_file}")
os.rename(temp_file, 'raw_data.csv')
os.remove("raw_data.tsv")
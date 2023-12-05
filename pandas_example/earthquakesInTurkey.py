# %%
import os

import pandas as pd
import requests

web_sitesi_adresi = "http://www.koeri.boun.edu.tr/sismo/trk.txt"
response = requests.get(url=web_sitesi_adresi)
response_content = response.content

print(response.encoding)
skip_line = 6
temp_file = 'dosya.txt'

with open(temp_file, 'wb') as f:
    f.write(response_content)

# %%
with open(temp_file, 'rb') as rf:
    for _ in range(skip_line):
        # 'utf-8' codec can't decode byte 0xdd in position 4: invalid continuation byte
        next(rf)

    df = pd.read_fwf(rf,
                     widths=(11, 10, 9, 10, 13, 16, 40),
                     names=('tarih', 'saat', 'enlem', 'boylam', 'derinlik', 'magnitud', 'yer'))

# if os.path.exists(temp_file):
#     os.remove(temp_file)
os.unlink(temp_file)

# %%
df
# %%
df.info()
# %%
yuksekDepremler = df['magnitud'] >= df['magnitud'].max()-1
df[yuksekDepremler]
# %%

# %%
import pandas as pd

web_sitesi_adresi = "http://www.koeri.boun.edu.tr/sismo/trk.txt"
df = pd.read_fwf(filepath_or_buffer=web_sitesi_adresi,
                 encoding='ISO-8859-9',
                 skiprows=[0, 1, 2, 3, 5, 6],
                 header=0)

df.info()

# %%
maxMag = df['Magnitud(Md)'].max()
# %%
# df2 = df[df['Magnitud(Md)'] > maxMag-1]
# df[df['Magnitud(Md)'] > maxMag-1].sort_values(by=["Magnitud(Md)"], ascending=False).head()

df.sort_values(by=['Magnitud(Md)'], ascending=False).head()

# %%
yeniListe = df.groupby(by=['Magnitud(Md)',
                           pd.to_datetime(df['Tarih'], format='%Y.%m.%d').dt.month]).size().unstack().fillna(0)
yeniListe
# %%

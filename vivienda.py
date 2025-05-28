# Pisos turisticos Euskadi COMPLETO
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

'''
ine_pisos_tur = pd.read_csv('./data/viviendas_turisticas_INE.csv', encoding='latin-1')


df = pd.read_csv("./data/viviendas_turisticas_INE.csv", sep=';', encoding='latin1')

df = df[df["Viviendas y plazas"] == "Viviendas turísticas"]

# Función para limpiar los valores numéricos que están mal escritos con . los miles
def limpiar_numero(valor):
    if pd.isna(valor):
        return None
    valor_str = str(valor).strip()
    valor_limpio = valor_str.replace('.', '')
    valor_limpio = valor_limpio.replace(' ', '')
    if valor_limpio == '' or valor_limpio == '-' or valor_limpio == '..':
        return None
    try:
        return int(valor_limpio)
    except ValueError:
        print(f"No se pudo convertir: '{valor}' -> '{valor_limpio}'")
        return None

df["Viviendas"] = df["Total"].apply(limpiar_numero)

df["Año"] = df["Periodo"].str[:4]

prov_df = df[df["Provincias"].notna()].copy()
prov_df["Entidad"] = prov_df["Provincias"].str.extract(r'\d{2}\s+(.*)')
prov_df["Entidad"] = prov_df["Entidad"].replace({
    "Araba/Álava": "Araba",
    "Bizkaia": "Bizkaia",
    "Gipuzkoa": "Gipuzkoa"
})

prov_max = prov_df.groupby(["Entidad", "Año"])["Viviendas"].max().reset_index()

cae_df = df[(df["Provincias"].isna()) & (df["Comunidades y Ciudades Autónomas"] == "16 País Vasco")]
cae_max = cae_df.groupby("Año")["Viviendas"].max().reset_index()
cae_max["Entidad"] = "CAE"

final_df = pd.concat([prov_max, cae_max], ignore_index=True)

df_pivot = final_df.pivot(index="Entidad", columns="Año", values="Viviendas")

df_pivot = df_pivot[sorted(df_pivot.columns, reverse=True)]

df_final = df_pivot.reset_index()
df_final.columns.name = None
df_final = df_final.rename(columns={"Entidad": "Territorio"})

df_final.to_csv("./data/viviendas_turisticas_v0.csv", sep=';', index=False)

'''


# Precio alquiler Euskadi - Territorios históricos COMPLETO


archivo = "./data/precio_alquiler.csv"

df = pd.read_csv(archivo, sep=';', skiprows=5, header=None, encoding='latin1')

territorios = ['C.A. de Euskadi', '   Araba/Álava', '   Bizkaia', '   Gipuzkoa']
df = df[df[0].isin(territorios)].reset_index(drop=True)

df[0] = df[0].str.strip().replace({
    'C.A. de Euskadi': 'CAE',
    'Araba/Álava': 'Araba',
    'Bizkaia': 'Bizkaia',
    'Gipuzkoa': 'Gipuzkoa'
})

df = df.dropna(axis=1, how='all')

data_only = df.iloc[:, 1:].applymap(lambda x: str(x).replace(',', '.')).astype(float)

años = list(range(2016, 2025))
n_trimestres = 4

columnas_anuales = {}
for i, año in enumerate(años):
    inicio = i * n_trimestres
    fin = inicio + n_trimestres
    columnas_anuales[año] = data_only.iloc[:, inicio:fin].mean(axis=1)

df_resultado = pd.DataFrame(columnas_anuales)
df_resultado.insert(0, 'Territorio', df[0])

df_resultado.to_csv('./data/precio_alquiler_.csv')

df_resultado



df = pd.read_csv("./data/precio_alquiler_.csv", sep=',', encoding="latin1")
df = df.drop(columns=df.columns[0])

df.to_csv('./data/precio_alquiler_final.csv', index=False)


# Precio compraventa libre - COMPLETO

def limpiar_csv_compraventa(ruta_csv, ruta_salida=None):
    
    df = pd.read_csv(ruta_csv, sep=';', skiprows=6, encoding='latin1', header=None)

    df.dropna(axis=1, how='all', inplace=True)

    columnas = ['Trimestre', 
                'Euskadi_Total', 'Euskadi_Nueva', 'Euskadi_Usada',
                'Álava_Total', 'Álava_Nueva', 'Álava_Usada',
                'Bizkaia_Total', 'Bizkaia_Nueva', 'Bizkaia_Usada',
                'Gipuzkoa_Total', 'Gipuzkoa_Nueva', 'Gipuzkoa_Usada']
    df.columns = columnas

    for col in columnas[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

        df.to_csv(ruta_salida, index=False)

    return df
limpiar_csv_compraventa('./data/precio.csv','./data/precio_cv_limpio.csv')
precio_cv_limpio = pd.read_csv('./data/precio_cv_limpio.csv')
precio_cv_limpio


# Función para cambiar los NaN de los años por la media de los trimestres y borrar los datos de los trimestres

def precio_año_media(df):
    df = df.copy()
    filas_a_conservar = []
    
    i = 0
    while i < len(df) - 4:
        fila = df.iloc[i]
        if str(fila['Trimestre']).isdigit() and pd.isna(fila[1:]).all():
            # Fila con año y NaNs: calcular media de los 4 trimestres siguientes
            medias = df.iloc[i+1:i+5].mean(numeric_only=True)
            fila_actualizada = fila.copy()
            fila_actualizada[1:] = medias.values
            filas_a_conservar.append(fila_actualizada)
            i += 5  # Saltar año + 4 trimestres
        else:
            i += 1  

    df_anual = pd.DataFrame(filas_a_conservar).reset_index(drop=True)
    return df_anual

df_venta_anual = precio_año_media(precio_cv_limpio)

df_venta_anual.head()

precio_venta_antestrasp = df_venta_anual.to_csv('./data/precio_venta_antestrasp.csv')

df = pd.read_csv('./data/precio_venta_antestrasp.csv', index_col=0)

cols_totales = [col for col in df.columns if col.endswith('_Total')]
cols_seleccionadas = ['Trimestre'] + cols_totales

df_totales = df[cols_seleccionadas].copy()

df_totales = df_totales.rename(columns={'Trimestre': 'Año'})


df = pd.read_csv('./data/precio_venta_antestrasp.csv', index_col=0)

#Para las medias seleccionamos las columnas
cols_totales = [col for col in df.columns if col.endswith('_Total')]
df_totales = df[['Trimestre'] + cols_totales]

df_totales = df_totales.rename(columns={'Trimestre': 'Año'})

df_transpuesto = df_totales.set_index('Año').T
df_transpuesto = df_transpuesto.rename(index={
    'Euskadi_Total': 'CAE',
    'Álava_Total': 'Araba',
    'Bizkaia_Total': 'Bizkaia',
    'Gipuzkoa_Total': 'Gipuzkoa'
})
df_transpuesto.columns = df_transpuesto.columns.astype(int)
df_transpuesto = df_transpuesto.rename(columns={2025: 2024})
df_transpuesto.index.name = 'Territorio'

df_transpuesto.to_csv('./data/precio_cv_final.csv')

# Licencias de vivienda nueva COMPLETO

archivo = './data/licencias.csv'

df = pd.read_csv(archivo, sep=';', header=None, encoding='latin1')

idx_inicio = df[df[0] == 'Entidad'].index[0]
idx_fin = df[df[0] == 'Gipuzkoa'].index[0]

df = df.loc[idx_inicio:idx_fin].reset_index(drop=True)

df = df.dropna(axis=1, how='all')

df = df.replace('"', '', regex=True)

df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

for col in df.columns[1:]:
    df[col] = df[col].str.replace(',', '.')
    df[col] = pd.to_numeric(df[col], errors='coerce')  

df = df.rename(columns={'Entidad': 'Territorio'})
df['Territorio'] = df['Territorio'].replace({'Araba/Álava': 'Araba'})

df.to_csv('./data/licencias_limpio.csv', index=False, sep=',')
#Esto sigue siendo cada 1000 habitantes

df_poblacion = pd.read_csv('./data/poblacion_total_limpio.csv')
df_licencias = pd.read_csv('./data/licencias_limpio.csv')

columnas_comunes = list(set(df_poblacion.columns[1:]) & set(df_licencias.columns[1:]))
columnas_comunes = sorted(columnas_comunes, reverse=True)

df_poblacion = df_poblacion[['Territorio'] + columnas_comunes].sort_values('Territorio').reset_index(drop=True)
df_licenciass = df_licencias[['Territorio'] + columnas_comunes].sort_values('Territorio').reset_index(drop=True)

df_totales = df_licencias.copy()
for col in columnas_comunes:
    df_totales[col] = ((df_licencias[col] / 1000) * df_poblacion[col]).round().astype('Int64')

df_totales.to_csv('./data/licencias_totales.csv', index=False)

df_totales.to_csv('./data/licencias_final.csv', index=False, sep=',')


# Suelo urbanizable COMPLETO

archivo = './data/suelo_urbanizable.csv'

df = pd.read_csv(archivo, sep=';', header=None, encoding='latin1')

idx_inicio = df[df[0] == 'Entidad'].index[0]
idx_fin = df[df[0] == 'Gipuzkoa'].index[0]

df = df.loc[idx_inicio:idx_fin].reset_index(drop=True)

df = df.dropna(axis=1, how='all')

df = df.replace('"', '', regex=True)

df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

for col in df.columns[1:]:
    df[col] = df[col].str.replace(',', '.')
    df[col] = pd.to_numeric(df[col], errors='coerce')  

df = df.rename(columns={'Entidad': 'Territorio'})
df['Territorio'] = df['Territorio'].replace({'Araba/Álava': 'Araba'})

df.to_csv('./data/suelo_urbanizable_final.csv', index=False, sep=',')


# Solicitudes de vivienda que constan en Etxebide ( x1000 habitantes) COMPLETO

archivo = './data/solicitudes_vivienda.csv'

df = pd.read_csv(archivo, sep=';', header=None, encoding='latin1')

idx_inicio = df[df[0] == 'Entidad'].index[0]
idx_fin = df[df[0] == 'Gipuzkoa'].index[0]

df = df.loc[idx_inicio:idx_fin].reset_index(drop=True)

df = df.dropna(axis=1, how='all')

df = df.replace('"', '', regex=True)

df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

for col in df.columns[1:]:
    df[col] = df[col].str.replace(',', '.')
    df[col] = pd.to_numeric(df[col], errors='coerce')  

df = df.rename(columns={'Entidad': 'Territorio'})
df['Territorio'] = df['Territorio'].replace({'Araba/Álava': 'Araba'})

df.to_csv('./data/solicitudes_vivienda_limpio.csv', index=False, sep=',')

#Esto sigue siendo cada 1000 habitantes

df_poblacion = pd.read_csv('./data/poblacion_total_limpio.csv')
df_solicitudes = pd.read_csv('./data/solicitudes_vivienda_limpio.csv')

columnas_comunes = list(set(df_poblacion.columns[1:]) & set(df_solicitudes.columns[1:]))
columnas_comunes = sorted(columnas_comunes, reverse=True)

df_poblacion = df_poblacion[['Territorio'] + columnas_comunes].sort_values('Territorio').reset_index(drop=True)
df_solicitudes = df_solicitudes[['Territorio'] + columnas_comunes].sort_values('Territorio').reset_index(drop=True)

df_totales = df_solicitudes.copy()
for col in columnas_comunes:
    df_totales[col] = ((df_solicitudes[col] / 1000) * df_poblacion[col]).round().astype('Int64')

df_totales.to_csv('./data/solicitantes_totales.csv', index=False)


# Viviendas adjudicadas por Etxebide en el ultimo trienio por cada 100 solicitudes inscritas  

archivo = './data/viviendas_etxebide.csv'

df = pd.read_csv(archivo, sep=';', header=None, encoding='latin1')

idx_inicio = df[df[0] == 'Entidad'].index[0]
idx_fin = df[df[0] == 'Gipuzkoa'].index[0]

df = df.loc[idx_inicio:idx_fin].reset_index(drop=True)

df = df.dropna(axis=1, how='all')

df = df.replace('"', '', regex=True)

df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

for col in df.columns[1:]:
    df[col] = df[col].str.replace(',', '.')
    df[col] = pd.to_numeric(df[col], errors='coerce')  

df = df.rename(columns={'Entidad': 'Territorio'})
df['Territorio'] = df['Territorio'].replace({'Araba/Álava': 'Araba'})

df.to_csv('./data/viviendas_etxebide_limpio.csv', index=False, sep=',')
#Esto sigue siendo cada 1000 habitantes

df_poblacion = pd.read_csv('./data/poblacion_total_limpio.csv')
df_etxebide = pd.read_csv('./data/viviendas_etxebide_limpio.csv')

columnas_comunes = list(set(df_poblacion.columns[1:]) & set(df_etxebide.columns[1:]))
columnas_comunes = sorted(columnas_comunes, reverse=True)

df_poblacion = df_poblacion[['Territorio'] + columnas_comunes].sort_values('Territorio').reset_index(drop=True)
df_etxebide = df_etxebide[['Territorio'] + columnas_comunes].sort_values('Territorio').reset_index(drop=True)

df_totales = df_etxebide.copy()
for col in columnas_comunes:
    df_totales[col] = ((df_etxebide[col] / 100) * df_poblacion[col]).round().astype('Int64')

df_totales.to_csv('./data/viviendas_etxebide_total.csv', index=False)

# Poblacion total (hab.) COMPLETO

archivo = './data/Poblacion_total.csv'

df = pd.read_csv(archivo, sep=';', header=None, encoding='latin1')

# Tabla entre 'Entidad' y 'Gipuzkoa'
idx_inicio = df[df[0] == 'Entidad'].index[0]
idx_fin = df[df[0] == 'Gipuzkoa'].index[0]
df = df.loc[idx_inicio:idx_fin].reset_index(drop=True)


df = df.dropna(axis=1, how='all')

df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

df = df.replace('"', '', regex=True)

for col in df.columns[1:]:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.strip().str.replace('.', '', regex=False)  # quitar miles
    df[col] = df[col].str.replace(',', '.', regex=False)             # decimal
    df[col] = pd.to_numeric(df[col], errors='coerce')                # convertir

# Se cambia para que tenga patrón común con el resto de csv, Territorio
df = df.rename(columns={'Entidad': 'Territorio'})
df['Territorio'] = df['Territorio'].replace({'Araba/Álava': 'Araba'})

df.to_csv('./data/poblacion_total_limpio.csv', index=False, sep=',')

# Población extranjera
archivo = './data/poblacion_extranjera.csv'

df = pd.read_csv(archivo, sep=';', header=None, encoding='latin1')

idx_inicio = df[df[0] == 'Entidad'].index[0]
idx_fin = df[df[0] == 'Gipuzkoa'].index[0]

df = df.loc[idx_inicio:idx_fin].reset_index(drop=True)

df = df.dropna(axis=1, how='all')

df = df.replace('"', '', regex=True)

df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

for col in df.columns[1:]:
    df[col] = df[col].str.replace(',', '.')
    df[col] = pd.to_numeric(df[col], errors='coerce')  

df = df.rename(columns={'Entidad': 'Territorio'})
df['Territorio'] = df['Territorio'].replace({'Araba/Álava': 'Araba'})

df.to_csv('./data/poblacion_extranjera_limpio.csv', index=False, sep=',')
#Esto sigue siendo cada 1000 habitantes

df_poblacion = pd.read_csv('./data/poblacion_total_limpio.csv')
df_extranjeros = pd.read_csv('./data/poblacion_extranjera_limpio.csv')

columnas_comunes = list(set(df_poblacion.columns[1:]) & set(df_extranjeros.columns[1:]))
columnas_comunes = sorted(columnas_comunes, reverse=True)

df_poblacion = df_poblacion[['Territorio'] + columnas_comunes].sort_values('Territorio').reset_index(drop=True)
df_extranjeross = df_extranjeros[['Territorio'] + columnas_comunes].sort_values('Territorio').reset_index(drop=True)

df_totales = df_extranjeros.copy()
for col in columnas_comunes:
    df_totales[col] = ((df_extranjeros[col] / 1000) * df_poblacion[col]).round().astype('Int64')

df_totales.to_csv('./data/poblacion_extranjera_total.csv', index=False)
# Mujeres electas en elecciones municipales (% sobre total personas electas).csv



archivo = './data/mujeres_electas.csv'

df = pd.read_csv(archivo, sep=';', header=None, encoding='latin1')

idx_inicio = df[df[0] == 'Entidad'].index[0]
idx_fin = df[df[0] == 'Gipuzkoa'].index[0]

df = df.loc[idx_inicio:idx_fin].reset_index(drop=True)

df = df.dropna(axis=1, how='all')

df = df.replace('"', '', regex=True)

df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

for col in df.columns[1:]:
    df[col] = df[col].str.replace(',', '.')
    df[col] = pd.to_numeric(df[col], errors='coerce')  

df = df.rename(columns={'Entidad': 'Territorio'})
df['Territorio'] = df['Territorio'].replace({'Araba/Álava': 'Araba'})

df.to_csv('./data/mujeres_limpio.csv', index=False, sep=',')

#Esto sigue siendo cada 1000 habitantes

df_poblacion = pd.read_csv('./data/poblacion_total_limpio.csv')
df_mujeres = pd.read_csv('./data/mujeres_limpio.csv')

columnas_comunes = list(set(df_poblacion.columns[1:]) & set(df_mujeres.columns[1:]))
columnas_comunes = sorted(columnas_comunes, reverse=True)

df_poblacion = df_poblacion[['Territorio'] + columnas_comunes].sort_values('Territorio').reset_index(drop=True)
df_mujeres = df_mujeres[['Territorio'] + columnas_comunes].sort_values('Territorio').reset_index(drop=True)

df_totales = df_mujeres.copy()
for col in columnas_comunes:
    df_totales[col] = ((df_mujeres[col] / 1000) * df_poblacion[col]).round().astype('Int64')

df_totales.to_csv('./data/solicitantes_totales.csv', index=False)



# --------------------------------------------------------------------------------
# -----          GRAFICOS --------------------------------------------------------
# --------------------------------------------------------------------------------



## Introducción - 
### I.1 Precio Compraventa - Territorio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

plt.rcdefaults()


df_precio = pd.read_csv('./data/precio_cv_final.csv')

df_prec_long = df_precio.melt(id_vars='Territorio', var_name='Año', value_name='Precio')

df_prec_long['Año'] = df_prec_long['Año'].astype(int)

# plt.rcdefaults()

plt.style.use("seaborn-v0_8")  # Un estilo tipo seaborn más suave

fig, ax1 = plt.subplots(figsize=(10, 6))

sns.lineplot(data=df_prec_long, x='Año', y='Precio', hue='Territorio', ax=ax1)
ax1.set_ylabel('Precio')
ax1.set_xlabel('Año')
ax1.set_title('Evolución precio compraventa vivienda según territorio y año')

# Formato del eje X: solo años enteros
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

plt.tight_layout()

# Código para ir repitiendo en cada gráfico, selecciona ruta salida
ruta = "./img/intro/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "precio_compraventa_territorio.png"
plt.savefig(os.path.join(ruta, nombre_archivo))

### I.2 Precio Alquiler - Teritorio - Año
import matplotlib.ticker as ticker

df_prec_alq = pd.read_csv('./data/precio_alquiler_final.csv')

df_precalq_long = df_prec_alq.melt(id_vars='Territorio', var_name='Año', value_name='Precio')

df_precalq_long['Año'] = df_precalq_long['Año'].astype(int)

fig, ax1 = plt.subplots(figsize=(10, 6))

sns.lineplot(data=df_precalq_long, x='Año', y='Precio', hue='Territorio', ax=ax1)
ax1.set_ylabel('Precio')
ax1.set_xlabel('Año')

#Sirve para que en eje X se vea como años enteros sin decimales
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))


plt.title('Evolución precio alquiler vivienda según territorio y año')
plt.tight_layout()

ruta = './img/intro/'
os.makedirs(ruta, exist_ok=True)
nombre_archivo = 'precio_alquiler_territorio_año.jpg'
plt.savefig(os.path.join(ruta, nombre_archivo))


## hipotesis 1: más vivienda en territorios con más suelo liberado



df_licencias = pd.read_csv('./data/licencias_totales.csv')
df_suelo = pd.read_csv('./data/suelo_urbanizable_final.csv')

df_lic_long = df_licencias.melt(id_vars='Territorio', var_name='Año', value_name='Licencias')
df_suelo_long = df_suelo.melt(id_vars='Territorio', var_name='Año', value_name='Suelo_urbanizable')

df_lic_long['Año'] = df_lic_long['Año'].astype(int)
df_suelo_long['Año'] = df_suelo_long['Año'].astype(int)

df_merged = pd.merge(df_lic_long, df_suelo_long, on=['Territorio', 'Año'])

# Grafico Scatter
plt.figure(figsize=(10, 6))
sns.set_style("white")
sns.scatterplot(data=df_merged, x='Suelo_urbanizable', y='Licencias', hue='Territorio')
plt.title('Relación entre suelo urbanizable y licencias de obra nueva')
plt.xlabel('Suelo urbanizable (%)')
plt.ylabel('Licencias')
plt.grid(True)
plt.tight_layout()

# Código para ir repitiendo en cada gráfico, selecciona ruta salida
ruta = "./img/hipotesis_1/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "relacion_suelo_licencias.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))


fig, ax1 = plt.subplots(figsize=(10, 6))

sns.lineplot(data=df_merged, x='Año', y='Licencias', hue='Territorio', ax=ax1)
ax1.set_ylabel('Licencias: Línea continua')
ax1.set_xlabel('Año')

#Sirve para que en eje X se vea como años enteros sin decimales
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

ax2 = ax1.twinx()
sns.lineplot(data=df_merged, x='Año', y='Suelo_urbanizable', hue='Territorio', ax=ax2, linestyle='--', legend=False)
ax2.set_ylabel('Suelo urbanizable (%) --')

plt.title('Evolución de licencias y suelo urbanizable por territorio')
plt.tight_layout()

ruta = "./img/hipotesis_1/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "licencias_suelo_territorio.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))


# Hipotesis 2 - mayor número de viviendas construidas baja el precio de la vivienda

df_precio = pd.read_csv('./data/precio_cv_final.csv')
df_licencias = pd.read_csv('./data/licencias_final.csv')

df_precio_melt = df_precio.melt(id_vars='Territorio', var_name='Año', value_name='Precio')
df_licencias_melt = df_licencias.melt(id_vars='Territorio', var_name='Año', value_name='Licencias')

df = pd.merge(df_precio_melt, df_licencias_melt, on=['Territorio', 'Año'])

df['Año'] = df['Año'].astype(int)



plt.figure(figsize=(10, 6))
sns.lmplot(data=df, x='Licencias', y='Precio', hue='Territorio', height=6, aspect=1.3)
plt.title('¿Mayor construcción implica menor precio?')
plt.xlabel('Nº de licencias de obra nueva')
plt.ylabel('Precio medio de compraventa (€)')
plt.grid(True)

# Código para ir repitiendo en cada gráfico, selecciona ruta salida
ruta = "./img/Hipotesis_2/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "construccion_y_precio.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))



df_lic = pd.read_csv('./data/licencias_final.csv')
df_lic_long = df_lic.melt(id_vars='Territorio', var_name='Año', value_name='Licencias')
df_lic_long['Año'] = df_lic_long['Año'].astype(int)

df_precio = pd.read_csv('./data/precio_cv_final.csv')
df_precio_long = df_precio.melt(id_vars='Territorio', var_name='Año', value_name='Precio_venta')
df_precio_long['Año'] = df_precio_long['Año'].astype(int)

df_merged = pd.merge(df_lic_long, df_precio_long, on=['Territorio', 'Año'])

correlations = df_merged.groupby('Territorio')[['Licencias', 'Precio_venta']].corr().iloc[0::2, 1]
pivot_corr = df_merged.pivot_table(index='Territorio', columns='Año', values='Precio_venta')
pivot_lic = df_merged.pivot_table(index='Territorio', columns='Año', values='Licencias')

correlacion = pivot_corr.corrwith(pivot_lic, axis=0)

plt.figure(figsize=(12, 1.5))
sns.heatmap([correlacion], cmap='coolwarm', annot=True, fmt=".2f", xticklabels=correlacion.index, yticklabels=['Correlación'])
plt.title('Correlación entre Licencias y Precio de Venta por Año')
plt.tight_layout()
ruta = "./img/hipotesis_2/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "relacion_licencias_precio_cv.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))


df_lic = pd.read_csv('./data/licencias_final.csv')
df_precio = pd.read_csv('./data/precio_cv_final.csv')

df_lic_long = df_lic.melt(id_vars='Territorio', var_name='Año', value_name='Licencias')
df_precio_long = df_precio.melt(id_vars='Territorio', var_name='Año', value_name='Precio_venta')

df_lic_long['Año'] = df_lic_long['Año'].astype(int)
df_precio_long['Año'] = df_precio_long['Año'].astype(int)

df = pd.merge(df_lic_long, df_precio_long, on=['Territorio', 'Año'])

correlaciones = df.groupby('Año').apply(lambda g: g['Licencias'].corr(g['Precio_venta'])).reset_index()
correlaciones.columns = ['Año', 'Correlación']

plt.figure(figsize=(10, 5))
sns.lineplot(data=correlaciones, x='Año', y='Correlación', marker='o')
plt.title('Correlación Licencias y Precio de Venta a lo largo del tiempo en Euskadi')
plt.ylim(-1, 1)
plt.axhline(0, color='green', linestyle='--')
plt.tight_layout()
ruta = "./img/hipotesis_2/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "correlacion_licencias_precio_cv.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))


df_licencias = pd.read_csv('./data/licencias_totales.csv')
df_precio = pd.read_csv('./data/precio_cv_final.csv')

df_lic_long = df_licencias.melt(id_vars='Territorio', var_name='Año', value_name='Licencias')
df_precio_long = df_precio.melt(id_vars='Territorio', var_name='Año', value_name='Precio_venta')

df_lic_long['Año'] = df_lic_long['Año'].astype(int)
df_precio_long['Año'] = df_precio_long['Año'].astype(int)

df = pd.merge(df_lic_long, df_precio_long, on=['Territorio', 'Año'])

correlaciones = (
    df.groupby('Territorio')[['Licencias', 'Precio_venta']]
    .corr()
    .iloc[0::2, -1]
    .reset_index()
    .rename(columns={'Precio_venta': 'Correlación'})[['Territorio', 'Correlación']]
)


plt.figure(figsize=(8, 5))
sns.barplot(data=correlaciones, x='Territorio', y='Correlación', palette='coolwarm', edgecolor='black')

plt.title('Correlación entre licencias y precio de compraventa por territorio')
plt.ylim(-1, 1)
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()

ruta = "./img/hipotesis_2/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "corelacion_licencias_precio_cv_territorio.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))


## Hipotesis 3 demanda no satisfecha de vivienda pública protegida y el aumento del precio del precio del alquiler
import pandas as pd

sol_df = pd.read_csv("./data/solicitantes_totales.csv")
entreg_df = pd.read_csv("./data/viviendas_etxebide_total.csv")
compra_df = pd.read_csv("./data/precio_cv_final.csv")
alquiler_df = pd.read_csv("./data/precio_alquiler_final.csv")

sol_long = sol_df.melt(id_vars="Territorio", var_name="Año", value_name="Solicitantes")
entreg_long = entreg_df.melt(id_vars="Territorio", var_name="Año", value_name="Viviendas_Entregadas")
compra_long = compra_df.melt(id_vars="Territorio", var_name="Año", value_name="Precio_Compraventa")
alquiler_long = alquiler_df.melt(id_vars="Territorio", var_name="Año", value_name="Precio_Alquiler")

sol_long["Año"] = sol_long["Año"].astype(str)
entreg_long["Año"] = entreg_long["Año"].astype(str)
compra_long["Año"] = compra_long["Año"].astype(str)
alquiler_long["Año"] = alquiler_long["Año"].astype(str)


df_merge_1 = pd.merge(sol_long, entreg_long, on=["Territorio", "Año"], how="outer")
df_merge_2 = pd.merge(df_merge_1, compra_long, on=["Territorio", "Año"], how="outer")
df_final = pd.merge(df_merge_2, alquiler_long, on=["Territorio", "Año"], how="outer")

cols_num = ["Solicitantes", "Viviendas_Entregadas", "Precio_Compraventa", "Precio_Alquiler"]
for col in cols_num:
    df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

df_final = df_final.dropna(how="all", subset=cols_num)


import seaborn as sns
import matplotlib.pyplot as plt

df_corr = df_final.dropna(subset=["Solicitantes", "Viviendas_Entregadas", "Precio_Compraventa", "Precio_Alquiler"])

corr_matrix = df_corr[["Solicitantes", "Viviendas_Entregadas", "Precio_Compraventa", "Precio_Alquiler"]].corr()

print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlación entre variables de vivienda")
plt.tight_layout()

ruta = "./img/hipotesis_3/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "corelacion_vpo_precios.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))

df_final["Demanda_No_Satisfecha"] = df_final["Solicitantes"] - df_final["Viviendas_Entregadas"]
df_final["Demanda_No_Satisfecha"] = df_final["Demanda_No_Satisfecha"].clip(lower=0)

df_test = df_final.dropna(subset=["Demanda_No_Satisfecha", "Precio_Alquiler"])

corr = df_test["Demanda_No_Satisfecha"].corr(df_test["Precio_Alquiler"])

sns.regplot(data=df_test, x="Demanda_No_Satisfecha", y="Precio_Alquiler")
plt.title("Demanda no satisfecha vs Precio alquiler")
plt.xlabel("Demanda no satisfecha (Solicitantes - Viviendas entregadas)")
plt.ylabel("Precio alquiler (€)")
plt.grid(True)
plt.tight_layout()
ruta = "./img/hipotesis_3/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "demanda_nosatisfecha_precio_alquiler.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))



## Hipotesis 4 Pisos turísticos


def cargar_csv_wide(path, nombre_variable):
    df = pd.read_csv(path, sep=",")
    df.columns = df.columns.str.strip()
    df = df.melt(id_vars="Territorio", var_name="Año", value_name=nombre_variable)
    df["Año"] = df["Año"].astype(int)
    df[nombre_variable] = df[nombre_variable].astype(float)
    return df

df_alquiler = cargar_csv_wide("./data/viviendas_alquiler.csv", "viviendas_alquiler")
df_turisticos = cargar_csv_wide("./data/viviendas_turisticas.csv", "pisos_turisticos")
df_precios = cargar_csv_wide("./data/precio_alquiler_final.csv", "precio_alquiler")

df = df_alquiler.merge(df_turisticos, on=["Territorio", "Año"], how="inner") \
                .merge(df_precios, on=["Territorio", "Año"], how="inner")

df["proporcion_turistico_alquiler"] = df["pisos_turisticos"] / df["viviendas_alquiler"]

territorios = df["Territorio"].unique()

for territorio in territorios:
    df_territorio = df[df["Territorio"] == territorio]
    cor = df_territorio[["pisos_turisticos", "viviendas_alquiler", "precio_alquiler", "proporcion_turistico_alquiler"]].corr()

    plt.figure(figsize=(6, 4))
    sns.heatmap(cor, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title(f"Pisos turísticos correlaciones: {territorio}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ruta = "./img/hipotesis_4/"
    os.makedirs(ruta, exist_ok=True)
    nombre_archivo = "pisos_turisticos_correlaciones.jpg"
    plt.savefig(os.path.join(ruta, nombre_archivo))

## Hipotesis 5 Ha aumentado el precio medio del metro cuadrado en la misma proporción que el precio de venta/alquiler del inmueble? 

df_m2 = pd.read_csv("./data/precio_m2_final.csv")
df_total = pd.read_csv("./data/precio_cv_final.csv")

df_m2_long = df_m2.melt(id_vars='Territorio', var_name='Año', value_name='Precio_m2')
df_total_long = df_total.melt(id_vars='Territorio', var_name='Año', value_name='Precio_total')

df_combined = pd.merge(df_m2_long, df_total_long, on=['Territorio', 'Año'])

df_combined['Superficie_estim_m2'] = df_combined['Precio_total'] / df_combined['Precio_m2']

df_combined = df_combined.sort_values(by=['Territorio', 'Año'])

df_combined['Año'] = df_combined['Año'].astype(int)

df_combined = df_combined.sort_values(['Territorio', 'Año'])

df_combined['Var_pct_m2'] = df_combined.groupby('Territorio')['Precio_m2'].pct_change() * 100
df_combined['Var_pct_total'] = df_combined.groupby('Territorio')['Precio_total'].pct_change() * 100

df_combined['Diferencia_pct'] = df_combined['Var_pct_m2'] - df_combined['Var_pct_total']

resumen = df_combined.groupby('Territorio')[['Var_pct_m2', 'Var_pct_total', 'Diferencia_pct']].mean().round(2)

df_combined.to_csv("./data/variaciones_precio_vs_total.csv", index=False)

# Melt para juntar ambas tasas en una sola columna para los graficos
df_plot = pd.melt(
    df_combined,
    id_vars=['Territorio', 'Año'],
    value_vars=['Var_pct_m2', 'Var_pct_total'],
    var_name='Tipo_variacion',
    value_name='Variacio%'
)

df_plot['Tipo_variacion'] = df_plot['Tipo_variacion'].map({
    'Var_pct_m2': 'Precio por m²',
    'Var_pct_total': 'Precio total inmueble'
})

g = sns.relplot(
    data=df_plot,
    x='Año',
    y='Variacio%',
    hue='Tipo_variacion',
    kind='line',
    col='Territorio',
    col_wrap=2,
    facet_kws={'sharey': False},
    height=4,
    aspect=1.3,
    marker='o'
)

g.set_titles("{col_name}")
g.fig.suptitle("Tasa de variación anual: Precio por m² vs Precio total", fontsize=16, y=1.05)
g.set_axis_labels("Año", "Variación %")
plt.xticks(rotation=45)
plt.tight_layout()

ruta = "./img/hipotesis_5/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "variacion_precio_m2_cv.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))


## Hipótesis 6 - Inmigración y solicitantes

sol_df = pd.read_csv("./data/solicitantes_totales.csv")
inm_df = pd.read_csv("./data/poblacion_extranjera_total.csv")

sol_long = sol_df.melt(id_vars="Territorio", var_name="Año", value_name="Solicitantes")
inm_long = inm_df.melt(id_vars="Territorio", var_name="Año", value_name="Inmigrantes")

sol_long["Año"] = sol_long["Año"].astype(str)
inm_long["Año"] = inm_long["Año"].astype(str)

merged = pd.merge(sol_long, inm_long, on=["Territorio", "Año"], how="inner")

merged = merged.dropna()

merged["Solicitantes"] = pd.to_numeric(merged["Solicitantes"], errors='coerce')
merged["Inmigrantes"] = pd.to_numeric(merged["Inmigrantes"], errors='coerce')

sns.lmplot(data=merged, x="Inmigrantes", y="Solicitantes", hue="Territorio", height=6, aspect=1.5)
plt.title("¿Más inmigrantes implica más solicitantes de vivienda?")
plt.grid(True)
plt.tight_layout()
ruta = "./img/hipotesis_6/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "relacion_inmigracion_solicitantes.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))



inm_df = pd.read_csv("./data/poblacion_extranjera_total.csv")
pob_df = pd.read_csv("./data/poblacion_total_limpio.csv")

inm_long = inm_df.melt(id_vars="Territorio", var_name="Año", value_name="Inmigrantes")
pob_long = pob_df.melt(id_vars="Territorio", var_name="Año", value_name="Poblacion")

merged = pd.merge(inm_long, pob_long, on=["Territorio", "Año"], how="inner")

merged["Inmigrantes"] = pd.to_numeric(merged["Inmigrantes"], errors='coerce')
merged["Poblacion"] = pd.to_numeric(merged["Poblacion"], errors='coerce')

merged = merged.dropna()

g = sns.lmplot(data=merged, x="Poblacion", y="Inmigrantes", hue="Territorio", height=6, aspect=1.5)

plt.title("Relación entre inmigración y población total")
plt.grid(True)
plt.tight_layout()

g._legend.set_bbox_to_anchor((0.1, 0.90)) 
g._legend.set_loc('upper left')

ruta = "./img/hipotesis_6/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "relacion_inmigracion_poblacion.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))



inm_df = pd.read_csv("./data/poblacion_extranjera_total.csv")
pob_df = pd.read_csv("./data/poblacion_total_limpio.csv")

inm_long = inm_df.melt(id_vars="Territorio", var_name="Año", value_name="Inmigrantes")
pob_long = pob_df.melt(id_vars="Territorio", var_name="Año", value_name="Poblacion")

merged = pd.merge(inm_long, pob_long, on=["Territorio", "Año"], how="inner")

merged["Inmigrantes"] = pd.to_numeric(merged["Inmigrantes"], errors='coerce')
merged["Poblacion"] = pd.to_numeric(merged["Poblacion"], errors='coerce')

merged = merged.dropna()

merged["Proporcion"] = merged["Inmigrantes"] / merged["Poblacion"]
merged["Porcentaje"] = merged["Proporcion"] * 100

plt.figure(figsize=(12, 6))
sns.barplot(data=merged, x="Año", y="Porcentaje", hue="Territorio", palette="bright",errorbar=None)
plt.title("Porcentaje de inmigrantes sobre población total por territorio")
plt.ylabel("Porcentaje (%)")
plt.xticks(rotation=45)
plt.tight_layout()

ruta = "./img/hipotesis_6/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "relacion_inmigrantes_poblacion_territorio.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))


inm_df = pd.read_csv("./data/poblacion_extranjera_total.csv")
pob_df = pd.read_csv("./data/poblacion_total_limpio.csv")


inm_long = inm_df.melt(id_vars="Territorio", var_name="Año", value_name="Inmigrantes")
pob_long = pob_df.melt(id_vars="Territorio", var_name="Año", value_name="Poblacion")


merged = pd.merge(inm_long, pob_long, on=["Territorio", "Año"], how="inner")

merged["Inmigrantes"] = pd.to_numeric(merged["Inmigrantes"], errors='coerce')
merged["Poblacion"] = pd.to_numeric(merged["Poblacion"], errors='coerce')


merged = merged.dropna()

max_val = max(merged["Inmigrantes"].max(), merged["Poblacion"].max())

g = sns.lmplot(data=merged, x="Poblacion", y="Inmigrantes", hue="Territorio", height=6, aspect=1.5)

#Establececemos los valores de los ejes para que se muestren en la misma
margin = max_val * 0.05 
g.set(xlim=(0, max_val + margin), ylim=(0, max_val + margin))


plt.title("Relación entre inmigración y población total")
plt.tight_layout()
plt.grid(True)
g._legend.set_bbox_to_anchor((0.1, 0.90))
g._legend.set_loc('upper left')

ruta = "./img/hipotesis_6/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "relacion_inmigracion_poblacion_mismaescala.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))

## Hipotesis 7 Un mayor porcentaje de mujeres electas en los gobiernos municipales se asocia con políticas de vivienda más activas o eficaces.


mujeres = pd.read_csv("./data/mujeres_limpio.csv")
licencias = pd.read_csv("./data/licencias_totales.csv")
suelo = pd.read_csv("./data/suelo_urbanizable_final.csv")

mujeres_long = mujeres.melt(id_vars="Territorio", var_name="Año", value_name="Mujeres_electas")
licencias_long = licencias.melt(id_vars="Territorio", var_name="Año", value_name="Licencias")
suelo_long = suelo.melt(id_vars="Territorio", var_name="Año", value_name="Suelo")

df = mujeres_long.merge(licencias_long, on=["Territorio", "Año"], how="inner")
df = df.merge(suelo_long, on=["Territorio", "Año"], how="left")

df["Año"] = df["Año"].astype(int)



plt.style.use("seaborn-v0_8")  # Un estilo tipo seaborn más suave



mujeres = pd.read_csv("./data/mujeres_limpio.csv")
licencias = pd.read_csv("./data/licencias_totales.csv")
suelo = pd.read_csv("./data/suelo_urbanizable_final.csv")

mujeres_long = mujeres.melt(id_vars="Territorio", var_name="Año", value_name="Mujeres_electas")
licencias_long = licencias.melt(id_vars="Territorio", var_name="Año", value_name="Licencias")
suelo_long = suelo.melt(id_vars="Territorio", var_name="Año", value_name="Suelo_urbanizable")

df = mujeres_long.merge(licencias_long, on=["Territorio", "Año"], how="inner")
df = df.merge(suelo_long, on=["Territorio", "Año"], how="left")

df["Año"] = df["Año"].astype(int)

corr_matrix = df[["Mujeres_electas", "Licencias", "Suelo_urbanizable"]].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlación entre mujeres electas, licencias y suelo urbanizable")
plt.tight_layout()

ruta = "./img/hipotesis_7/"
os.makedirs(ruta, exist_ok=True)
nombre_archivo = "relacion_mujeres_licencias.jpg"
plt.savefig(os.path.join(ruta, nombre_archivo))

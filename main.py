## Pisos turisticos Euskadi COMPLETO
import numpy as np
import pandas as pd

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

df_final.to_csv("./data/viviendas_turisticas.csv", sep=';', index=False)
## Precio alquiler Euskadi - Territorios históricos COMPLETO

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

# Columnas: 'Trimestre' + las que terminan en _Total
cols_totales = [col for col in df.columns if col.endswith('_Total')]
cols_seleccionadas = ['Trimestre'] + cols_totales
df_totales = df_totales.rename(columns={'Trimestre': 'Año'})
df_totales = df[cols_seleccionadas]

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

print(df_totales.head())

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

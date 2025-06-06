import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil

# Cargar el CSV con metadatos
df = pd.read_csv('ddi_metadata.csv')

# Crear columna binaria 'binary_label'
df['binary_label'] = df['malignant'].apply(lambda x: 'malign' if x else 'benign')


# Si quieres ver el recuento por diagnóstico:
print("\nConteo por diagnóstico SIN undersampling :")
print(df['disease'].value_counts())

# Configurar estilo
sns.set(style="whitegrid")

# Crear figura con 1 subgráfico
fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # usa 'ax' en singular

# Gráfico de barras para benign/malign
sns.countplot(ax=ax, data=df, x='binary_label', hue='binary_label',
              order=['benign', 'malign'], palette='Set2', legend=False)

# Títulos y etiquetas
ax.set_title('Distribució Binària (benign vs malign)')
ax.set_xlabel('Classe')
ax.set_ylabel('Quantitat')

# Ajustar espaciado y mostrar
plt.tight_layout()
plt.show()

# Verificar el conteo de clases
print("\nConteo de clases SIN Undersampling (dx):")
print(df['binary_label'].value_counts())

# Número de muestras de la clase minoritaria
min_count = df['binary_label'].value_counts().min()

# Hacer undersampling de la clase mayoritaria
df_benign = df[df['binary_label'] == 'benign'].sample(n=min_count, random_state=42)
df_malign = df[df['binary_label'] == 'malign']

# Concatenar para obtener el dataset balanceado
df_balanced = pd.concat([df_benign, df_malign])

# Barajar el dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

df['undersample_selected'] = df['DDI_file'].isin(df_balanced['DDI_file'])

# Filtrar las filas seleccionadas por undersampling
df_selected = df[df['undersample_selected'] == True]

# Verificar el conteo de clases con undersampling
print("\nConteo de clases con Undersampling (dx):")
print(df_selected['binary_label'].value_counts())

# Si quieres ver el recuento por diagnóstico:
print("\nConteo por diagnóstico con Undersampling (dx):")
print(df_selected['disease'].value_counts())

# Configurar estilo
sns.set(style="whitegrid")

# Crear figura con 1 subgráfico
fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # usa 'ax' en singular

# Gráfico de barras para benign/malign
sns.countplot(ax=ax, data=df_selected, x='binary_label', hue='binary_label',
              order=['benign', 'malign'], palette='Set2', legend=False)

# Títulos y etiquetas
ax.set_title('Distribució Binària AMB UNDERSAMPLING (benign vs malign)')
ax.set_xlabel('Classe')
ax.set_ylabel('Quantitat')

# Ajustar espaciado y mostrar
plt.tight_layout()
plt.show()

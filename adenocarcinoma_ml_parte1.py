# PARTE 1: Generación de datos y análisis exploratorio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Generar datos
n_sanos = 250
n_cancer = 250

# Biomarcadores sanos
cea_sano = np.random.normal(2.5, 1.0, n_sanos)
ca19_9_sano = np.random.normal(15, 8, n_sanos)
ca125_sano = np.random.normal(20, 10, n_sanos)
afp_sano = np.random.normal(5, 2, n_sanos)
gene1_sano = np.random.normal(5, 1.5, n_sanos)
gene2_sano = np.random.normal(4, 1.2, n_sanos)
gene3_sano = np.random.normal(6, 1.8, n_sanos)

# Biomarcadores cáncer
cea_cancer = np.random.normal(25, 15, n_cancer)
ca19_9_cancer = np.random.normal(120, 80, n_cancer)
ca125_cancer = np.random.normal(85, 50, n_cancer)
afp_cancer = np.random.normal(30, 20, n_cancer)
gene1_cancer = np.random.normal(15, 5, n_cancer)
gene2_cancer = np.random.normal(2, 1, n_cancer)
gene3_cancer = np.random.normal(12, 4, n_cancer)

# Crear DataFrame
df = pd.DataFrame({
    'CEA': np.abs(np.concatenate([cea_sano, cea_cancer])),
    'CA19_9': np.abs(np.concatenate([ca19_9_sano, ca19_9_cancer])),
    'CA125': np.abs(np.concatenate([ca125_sano, ca125_cancer])),
    'AFP': np.abs(np.concatenate([afp_sano, afp_cancer])),
    'GeneExpr1': np.concatenate([gene1_sano, gene1_cancer]),
    'GeneExpr2': np.concatenate([gene2_sano, gene2_cancer]),
    'GeneExpr3': np.concatenate([gene3_sano, gene3_cancer]),
    'Edad': np.concatenate([
        np.random.choice(['<40', '40-60', '>60'], n_sanos, p=[0.3, 0.4, 0.3]),
        np.random.choice(['<40', '40-60', '>60'], n_cancer, p=[0.1, 0.3, 0.6])
    ]),
    'Sexo': np.random.choice(['M', 'F'], n_sanos+n_cancer, p=[0.48, 0.52]),
    'Fumador': np.concatenate([
        np.random.choice(['No', 'Ex-fumador', 'Fumador'], n_sanos, p=[0.6, 0.25, 0.15]),
        np.random.choice(['No', 'Ex-fumador', 'Fumador'], n_cancer, p=[0.3, 0.35, 0.35])
    ]),
    'Antecedentes_Familiares': np.concatenate([
        np.random.choice(['No', 'Sí'], n_sanos, p=[0.85, 0.15]),
        np.random.choice(['No', 'Sí'], n_cancer, p=[0.6, 0.4])
    ]),
    'Estadio': np.concatenate([
        np.array(['Sano']*n_sanos),
        np.random.choice(['I', 'II', 'III', 'IV'], n_cancer, p=[0.2, 0.3, 0.3, 0.2])
    ]),
    'Diagnostico': np.array(['Sano']*n_sanos + ['Adenocarcinoma']*n_cancer)
})

# Guardar datos
df.to_csv('adenocarcinoma_dataset.csv', index=False)
print("Dataset guardado: adenocarcinoma_dataset.csv")
print(f"Shape: {df.shape}")
print(f"\n{df.head()}")

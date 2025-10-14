# 🚀 Guía de Inicio Rápido

## Configuración Inicial

### 1. Instalar Dependencias

Abre una terminal en esta carpeta y ejecuta:

```bash
pip install -r requirements.txt
```

### 2. Iniciar Jupyter Notebook

```bash
jupyter notebook
```

Esto abrirá Jupyter en tu navegador web.

### 3. Abrir el Notebook

En la interfaz de Jupyter, haz clic en:
- `adenocarcinoma_ml_analysis.ipynb`

### 4. Ejecutar el Análisis

**Opción A - Ejecutar todo:**
- Menú: `Cell` → `Run All`

**Opción B - Paso a paso:**
- Presiona `Shift + Enter` en cada celda para ejecutarla

## 📊 Estructura del Notebook

### Sección 1: Generación de Datos
- Crea 500 pacientes simulados (250 sanos, 250 con cáncer)
- 7 biomarcadores + 4 variables clínicas

### Sección 2: Análisis Exploratorio
- Histogramas de biomarcadores
- Box plots comparativos
- Matriz de correlación
- Pair plots
- Análisis de variables clínicas
- Distribución de estadios

### Sección 3: Preparación de Datos
- División train/test (75/25)
- Normalización de biomarcadores
- Codificación de variables categóricas

### Sección 4: Modelos de ML
Entrena 4 modelos:
1. Regresión Logística
2. Random Forest
3. Gradient Boosting
4. SVM

### Sección 5: Comparación de Modelos
- Tabla comparativa de métricas
- Matrices de confusión
- Curvas ROC
- Curvas Precision-Recall

### Sección 6: Importancia de Características
- Ranking de biomarcadores más importantes
- Coeficientes de modelos lineales

### Sección 7: Conclusiones
- Resumen de hallazgos
- Recomendaciones

## 🎯 Resultados Esperados

Al ejecutar el notebook, verás:

✓ **20+ visualizaciones** de alta calidad
✓ **Métricas detalladas** de cada modelo
✓ **AUC-ROC > 0.90** en los mejores modelos
✓ **Interpretación** de características importantes

## 💡 Tips

- **Tiempo de ejecución:** ~2-3 minutos
- **Gráficas interactivas:** Todas las gráficas se generan automáticamente
- **Datos reproducibles:** Semilla fija (42) garantiza resultados consistentes
- **Experimentación:** Puedes modificar parámetros y re-ejecutar

## 🔧 Personalización

### Cambiar tamaño del dataset:
Modifica en la Sección 1:
```python
n_sanos, n_cancer = 250, 250  # Cambiar estos valores
```

### Ajustar modelos:
Modifica hiperparámetros en la Sección 4:
```python
rf = RandomForestClassifier(
    n_estimators=100,  # Número de árboles
    max_depth=10,      # Profundidad máxima
    random_state=42
)
```

### Agregar más biomarcadores:
En la Sección 1, añade nuevas columnas al DataFrame.

## ❓ Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
# Instalar librería faltante
pip install <nombre_libreria>
```

### Error: "seaborn-v0_8-darkgrid not found"
```python
# Cambiar en el notebook:
plt.style.use('seaborn-darkgrid')  # o 'ggplot'
```

### Gráficas no se muestran:
```python
# Asegúrate de tener esta línea:
%matplotlib inline
```

## 📁 Archivos Generados

Durante la ejecución, el notebook genera:
- Todas las visualizaciones en línea
- No se guardan archivos adicionales (opcional: puedes exportar)

## 📤 Exportar Resultados

### Guardar como HTML:
```bash
jupyter nbconvert --to html adenocarcinoma_ml_analysis.ipynb
```

### Guardar como PDF:
```bash
jupyter nbconvert --to pdf adenocarcinoma_ml_analysis.ipynb
```

### Guardar datos generados:
Añade al final del notebook:
```python
df.to_csv('datos_adenocarcinoma.csv', index=False)
```

## 🎓 Aprendizaje

Este notebook cubre:
- ✅ Generación de datos sintéticos
- ✅ Análisis exploratorio de datos (EDA)
- ✅ Preprocesamiento de datos
- ✅ Entrenamiento de modelos ML
- ✅ Evaluación de modelos
- ✅ Visualización de resultados
- ✅ Interpretabilidad de modelos

## 📚 Recursos Adicionales

- **Scikit-learn:** https://scikit-learn.org/
- **Pandas:** https://pandas.pydata.org/
- **Seaborn:** https://seaborn.pydata.org/
- **Matplotlib:** https://matplotlib.org/

## 🤝 Soporte

Si encuentras problemas:
1. Verifica que todas las librerías estén instaladas
2. Reinicia el kernel: `Kernel` → `Restart & Clear Output`
3. Ejecuta las celdas en orden

---

**¡Disfruta explorando el análisis de Machine Learning!** 🚀

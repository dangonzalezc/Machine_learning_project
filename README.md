# Machine Learning para Clasificación de Adenocarcinoma

## 📋 Descripción del Proyecto

Este proyecto implementa un análisis completo de **Machine Learning** para diferenciar pacientes con **adenocarcinoma** de individuos sanos, utilizando biomarcadores (variables continuas) y variables clínicas (categóricas).

## 🎯 Objetivos

- Generar datos simulados realistas de pacientes
- Realizar análisis exploratorio de datos (EDA)
- Entrenar y evaluar múltiples modelos de clasificación
- Comparar el desempeño de diferentes algoritmos
- Visualizar resultados con gráficas de ciencia de datos

## 📊 Dataset Simulado

### Características del Dataset:
- **500 pacientes totales**
  - 250 individuos sanos
  - 250 con adenocarcinoma

### Variables Continuas (Biomarcadores):
1. **CEA** (Antígeno Carcinoembrionario)
2. **CA19-9** (Marcador tumoral)
3. **CA125** (Marcador tumoral)
4. **AFP** (Alfa-fetoproteína)
5. **GeneExpr1, GeneExpr2, GeneExpr3** (Expresión génica)

### Variables Categóricas (Clínicas):
1. **Edad**: <40, 40-60, >60 años
2. **Sexo**: M, F
3. **Fumador**: No, Ex-fumador, Fumador
4. **Antecedentes Familiares**: Sí, No
5. **Estadio**: I, II, III, IV (solo para pacientes con cáncer)

## 🤖 Modelos de Machine Learning

El proyecto implementa y compara 4 algoritmos:

1. **Regresión Logística**
2. **Random Forest**
3. **Gradient Boosting**
4. **Support Vector Machine (SVM)**

## 📈 Visualizaciones Incluidas

### Análisis Exploratorio:
- Histogramas de distribución de biomarcadores
- Box plots comparativos
- Matriz de correlación
- Pair plots
- Distribución de variables clínicas
- Gráfica de estadios (pie chart)

### Evaluación de Modelos:
- Comparación de métricas (Accuracy, Precision, Recall, F1, AUC-ROC)
- Matrices de confusión
- Curvas ROC
- Curvas Precision-Recall
- Importancia de características
- Coeficientes de modelos lineales

## 🚀 Cómo Usar

### Requisitos:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Ejecutar el Notebook:
1. Abrir Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Abrir el archivo: `adenocarcinoma_ml_analysis.ipynb`

3. Ejecutar todas las celdas en orden

## 📁 Archivos del Proyecto

- `adenocarcinoma_ml_analysis.ipynb` - Cuaderno principal con todo el análisis
- `crear_notebook_final.py` - Script para generar el notebook
- `adenocarcinoma_ml_parte1.py` - Script auxiliar para generación de datos
- `README.md` - Este archivo

## 📊 Métricas de Evaluación

Los modelos son evaluados usando:

- **Accuracy**: Precisión general
- **Precision**: Proporción de verdaderos positivos
- **Recall (Sensibilidad)**: Capacidad de detectar casos positivos
- **F1-Score**: Media armónica de Precision y Recall
- **AUC-ROC**: Área bajo la curva ROC
- **Matriz de Confusión**: Análisis detallado de predicciones

## 🎓 Conceptos Aplicados

### Ciencia de Datos:
- Análisis exploratorio de datos (EDA)
- Visualización de datos
- Preprocesamiento (normalización, codificación)
- Feature engineering

### Machine Learning:
- Clasificación binaria
- Validación de modelos
- Métricas de evaluación
- Comparación de algoritmos
- Interpretabilidad de modelos

### Biomedicina:
- Biomarcadores tumorales
- Variables clínicas
- Estadios del cáncer
- Factores de riesgo

## 🔬 Aplicaciones Potenciales

1. **Diagnóstico Asistido**: Apoyo en la detección temprana
2. **Estratificación de Riesgo**: Identificación de pacientes de alto riesgo
3. **Biomarcadores**: Evaluación de la utilidad de diferentes marcadores
4. **Investigación Clínica**: Base para estudios prospectivos

## ⚠️ Nota Importante

Este proyecto utiliza **datos simulados** con fines educativos. Para aplicaciones clínicas reales, se requiere:

- Validación con datos reales de pacientes
- Aprobación de comités de ética
- Validación clínica rigurosa
- Cumplimiento de regulaciones médicas

## 📚 Referencias

- Biomarcadores tumorales en adenocarcinoma
- Algoritmos de Machine Learning para diagnóstico médico
- Métricas de evaluación en clasificación médica

## 👨‍💻 Autor

Proyecto desarrollado por Daniel González con el apoyo del modelo Claude Sonnet 4.5 Thinking en Windsurf y Google Colab como parte del Bootcamp de Inteligencia Artificial del programa TALENTO TECH del Ministerio de Tecnologías de la Información y las Comunicaciones de Colombia

## 📝 Licencia

Este proyecto es de uso educativo y académico.

---

**¡Explora el notebook para ver análisis detallados y visualizaciones!** 🚀

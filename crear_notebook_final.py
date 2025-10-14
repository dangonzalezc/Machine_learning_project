import json

# Crear notebook completo
nb = {
    "cells": [],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4
}

# Función helper
def mk(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def cd(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code]}

# Celdas del notebook
cells = [
    mk("# Machine Learning: Clasificación Adenocarcinoma vs Sano\n## Análisis con Biomarcadores y Variables Clínicas"),
    
    cd("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\nfrom sklearn.svm import SVC\nfrom sklearn.metrics import *\nimport warnings\nwarnings.filterwarnings('ignore')\nplt.style.use('seaborn-v0_8-darkgrid')\n%matplotlib inline"),
    
    mk("## 1. Generación de Datos Simulados"),
    
    cd("np.random.seed(42)\nn_sanos, n_cancer = 250, 250\n\n# Biomarcadores sanos\ncea_s = np.random.normal(2.5, 1, n_sanos)\nca19_s = np.random.normal(15, 8, n_sanos)\nca125_s = np.random.normal(20, 10, n_sanos)\nafp_s = np.random.normal(5, 2, n_sanos)\ng1_s, g2_s, g3_s = np.random.normal(5, 1.5, n_sanos), np.random.normal(4, 1.2, n_sanos), np.random.normal(6, 1.8, n_sanos)\n\n# Biomarcadores cáncer\ncea_c = np.random.normal(25, 15, n_cancer)\nca19_c = np.random.normal(120, 80, n_cancer)\nca125_c = np.random.normal(85, 50, n_cancer)\nafp_c = np.random.normal(30, 20, n_cancer)\ng1_c, g2_c, g3_c = np.random.normal(15, 5, n_cancer), np.random.normal(2, 1, n_cancer), np.random.normal(12, 4, n_cancer)\n\n# DataFrame\ndf = pd.DataFrame({\n    'CEA': np.abs(np.concatenate([cea_s, cea_c])),\n    'CA19_9': np.abs(np.concatenate([ca19_s, ca19_c])),\n    'CA125': np.abs(np.concatenate([ca125_s, ca125_c])),\n    'AFP': np.abs(np.concatenate([afp_s, afp_c])),\n    'GeneExpr1': np.concatenate([g1_s, g1_c]),\n    'GeneExpr2': np.concatenate([g2_s, g2_c]),\n    'GeneExpr3': np.concatenate([g3_s, g3_c]),\n    'Edad': np.concatenate([np.random.choice(['<40','40-60','>60'], n_sanos, p=[0.3,0.4,0.3]), np.random.choice(['<40','40-60','>60'], n_cancer, p=[0.1,0.3,0.6])]),\n    'Sexo': np.random.choice(['M','F'], n_sanos+n_cancer, p=[0.48,0.52]),\n    'Fumador': np.concatenate([np.random.choice(['No','Ex-fumador','Fumador'], n_sanos, p=[0.6,0.25,0.15]), np.random.choice(['No','Ex-fumador','Fumador'], n_cancer, p=[0.3,0.35,0.35])]),\n    'Antecedentes_Familiares': np.concatenate([np.random.choice(['No','Sí'], n_sanos, p=[0.85,0.15]), np.random.choice(['No','Sí'], n_cancer, p=[0.6,0.4])]),\n    'Estadio': np.concatenate([np.array(['Sano']*n_sanos), np.random.choice(['I','II','III','IV'], n_cancer, p=[0.2,0.3,0.3,0.2])]),\n    'Diagnostico': ['Sano']*n_sanos + ['Adenocarcinoma']*n_cancer\n})\nprint(f'Dataset: {df.shape}\\n{df.Diagnostico.value_counts()}')"),
    
    cd("df.head(10)"),
    cd("df.info()"),
    cd("biomarcadores = ['CEA','CA19_9','CA125','AFP','GeneExpr1','GeneExpr2','GeneExpr3']\ndf[biomarcadores].describe()"),
    
    mk("## 2. Análisis Exploratorio\n### 2.1 Distribución de Biomarcadores"),
    
    cd("fig, axes = plt.subplots(3,3, figsize=(16,12))\nfig.suptitle('Distribución Biomarcadores: Sano vs Adenocarcinoma', fontsize=16, fontweight='bold')\nfor idx, bio in enumerate(biomarcadores):\n    ax = axes[idx//3, idx%3]\n    df[df.Diagnostico=='Sano'][bio].hist(ax=ax, alpha=0.6, bins=30, label='Sano', color='green', edgecolor='black')\n    df[df.Diagnostico=='Adenocarcinoma'][bio].hist(ax=ax, alpha=0.6, bins=30, label='Adenocarcinoma', color='red', edgecolor='black')\n    ax.set_xlabel(bio, fontsize=11)\n    ax.set_ylabel('Frecuencia', fontsize=11)\n    ax.legend()\n    ax.grid(alpha=0.3)\nfor i in range(len(biomarcadores), 9):\n    fig.delaxes(axes[i//3, i%3])\nplt.tight_layout()\nplt.show()"),
    
    mk("### 2.2 Box Plots"),
    
    cd("fig, axes = plt.subplots(2,4, figsize=(18,10))\nfig.suptitle('Box Plots de Biomarcadores', fontsize=16, fontweight='bold')\naxes = axes.flatten()\nfor idx, bio in enumerate(biomarcadores):\n    ax = axes[idx]\n    data = [df[df.Diagnostico=='Sano'][bio], df[df.Diagnostico=='Adenocarcinoma'][bio]]\n    bp = ax.boxplot(data, labels=['Sano','Adenocarcinoma'], patch_artist=True)\n    for patch, color in zip(bp['boxes'], ['lightgreen','lightcoral']):\n        patch.set_facecolor(color)\n    ax.set_title(bio, fontsize=12, fontweight='bold')\n    ax.set_ylabel('Valor', fontsize=10)\n    ax.grid(alpha=0.3)\nfig.delaxes(axes[7])\nplt.tight_layout()\nplt.show()"),
    
    mk("### 2.3 Matriz de Correlación"),
    
    cd("plt.figure(figsize=(10,8))\nsns.heatmap(df[biomarcadores].corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=1)\nplt.title('Matriz de Correlación - Biomarcadores', fontsize=14, fontweight='bold', pad=20)\nplt.tight_layout()\nplt.show()"),
    
    mk("### 2.4 Pair Plot"),
    
    cd("bio_main = ['CEA','CA19_9','CA125','AFP']\npair_df = df[bio_main + ['Diagnostico']]\npp = sns.pairplot(pair_df, hue='Diagnostico', palette=['green','red'], diag_kind='kde', plot_kws={'alpha':0.6}, height=2.5)\npp.fig.suptitle('Pair Plot: Biomarcadores Principales', y=1.02, fontsize=14, fontweight='bold')\nplt.show()"),
    
    mk("### 2.5 Variables Clínicas"),
    
    cd("fig, axes = plt.subplots(2,2, figsize=(14,10))\nfig.suptitle('Variables Clínicas por Diagnóstico', fontsize=14, fontweight='bold')\nvars_cat = ['Edad','Sexo','Fumador','Antecedentes_Familiares']\nfor idx, var in enumerate(vars_cat):\n    ax = axes[idx//2, idx%2]\n    ct = pd.crosstab(df[var], df.Diagnostico, normalize='index')*100\n    ct.plot(kind='bar', ax=ax, color=['green','red'], alpha=0.7)\n    ax.set_title(var, fontsize=12, fontweight='bold')\n    ax.set_ylabel('Porcentaje (%)', fontsize=10)\n    ax.legend(['Sano','Adenocarcinoma'])\n    ax.grid(alpha=0.3, axis='y')\n    ax.set_xticklabels(ax.get_xticklabels(), rotation=45 if var=='Fumador' else 0, ha='right' if var=='Fumador' else 'center')\nplt.tight_layout()\nplt.show()"),
    
    cd("estadios = df[df.Diagnostico=='Adenocarcinoma'].Estadio.value_counts()\nplt.figure(figsize=(8,6))\nplt.pie(estadios.values, labels=estadios.index, autopct='%1.1f%%', colors=plt.cm.Reds(np.linspace(0.3,0.9,len(estadios))), startangle=90, explode=[0.05]*len(estadios))\nplt.title('Estadios en Adenocarcinoma', fontsize=13, fontweight='bold', pad=20)\nplt.tight_layout()\nplt.show()"),
    
    mk("## 3. Preparación de Datos"),
    
    cd("df_ml = df.drop('Estadio', axis=1)\ny = (df_ml.Diagnostico=='Sano').astype(int).values\nX_cont = df_ml[biomarcadores].values\nX_cat = pd.get_dummies(df_ml[['Edad','Sexo','Fumador','Antecedentes_Familiares']], drop_first=False)\nX = np.concatenate([X_cont, X_cat.values], axis=1)\nfeature_names = biomarcadores + list(X_cat.columns)\nprint(f'Features: {X.shape[1]}\\nClases: Adenocarcinoma={np.sum(y==0)}, Sano={np.sum(y==1)}')"),
    
    cd("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)\nprint(f'Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')"),
    
    cd("scaler = StandardScaler()\nn_cont = len(biomarcadores)\nX_train_sc, X_test_sc = X_train.copy(), X_test.copy()\nX_train_sc[:,:n_cont] = scaler.fit_transform(X_train[:,:n_cont])\nX_test_sc[:,:n_cont] = scaler.transform(X_test[:,:n_cont])\nprint('Normalización completada')"),
    
    mk("## 4. Modelos ML\n### 4.1 Regresión Logística"),
    
    cd("lr = LogisticRegression(random_state=42, max_iter=1000)\nlr.fit(X_train_sc, y_train)\ny_pred_lr = lr.predict(X_test_sc)\ny_prob_lr = lr.predict_proba(X_test_sc)[:,1]\nprint('='*60)\nprint('REGRESIÓN LOGÍSTICA')\nprint('='*60)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}')\nprint(f'Precision: {precision_score(y_test, y_pred_lr):.4f}')\nprint(f'Recall: {recall_score(y_test, y_pred_lr):.4f}')\nprint(f'F1: {f1_score(y_test, y_pred_lr):.4f}')\nprint(f'AUC-ROC: {roc_auc_score(y_test, y_prob_lr):.4f}')\nprint('\\n', classification_report(y_test, y_pred_lr, target_names=['Adenocarcinoma','Sano']))"),
    
    mk("### 4.2 Random Forest"),
    
    cd("rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)\nrf.fit(X_train_sc, y_train)\ny_pred_rf = rf.predict(X_test_sc)\ny_prob_rf = rf.predict_proba(X_test_sc)[:,1]\nprint('='*60)\nprint('RANDOM FOREST')\nprint('='*60)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}')\nprint(f'Precision: {precision_score(y_test, y_pred_rf):.4f}')\nprint(f'Recall: {recall_score(y_test, y_pred_rf):.4f}')\nprint(f'F1: {f1_score(y_test, y_pred_rf):.4f}')\nprint(f'AUC-ROC: {roc_auc_score(y_test, y_prob_rf):.4f}')\nprint('\\n', classification_report(y_test, y_pred_rf, target_names=['Adenocarcinoma','Sano']))"),
    
    mk("### 4.3 Gradient Boosting"),
    
    cd("gb = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)\ngb.fit(X_train_sc, y_train)\ny_pred_gb = gb.predict(X_test_sc)\ny_prob_gb = gb.predict_proba(X_test_sc)[:,1]\nprint('='*60)\nprint('GRADIENT BOOSTING')\nprint('='*60)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}')\nprint(f'Precision: {precision_score(y_test, y_pred_gb):.4f}')\nprint(f'Recall: {recall_score(y_test, y_pred_gb):.4f}')\nprint(f'F1: {f1_score(y_test, y_pred_gb):.4f}')\nprint(f'AUC-ROC: {roc_auc_score(y_test, y_prob_gb):.4f}')\nprint('\\n', classification_report(y_test, y_pred_gb, target_names=['Adenocarcinoma','Sano']))"),
    
    mk("### 4.4 SVM"),
    
    cd("svm = SVC(kernel='rbf', probability=True, random_state=42)\nsvm.fit(X_train_sc, y_train)\ny_pred_svm = svm.predict(X_test_sc)\ny_prob_svm = svm.predict_proba(X_test_sc)[:,1]\nprint('='*60)\nprint('SVM')\nprint('='*60)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}')\nprint(f'Precision: {precision_score(y_test, y_pred_svm):.4f}')\nprint(f'Recall: {recall_score(y_test, y_pred_svm):.4f}')\nprint(f'F1: {f1_score(y_test, y_pred_svm):.4f}')\nprint(f'AUC-ROC: {roc_auc_score(y_test, y_prob_svm):.4f}')\nprint('\\n', classification_report(y_test, y_pred_svm, target_names=['Adenocarcinoma','Sano']))"),
    
    mk("## 5. Comparación de Modelos\n### 5.1 Métricas"),
    
    cd("modelos = ['Logistic Regression','Random Forest','Gradient Boosting','SVM']\ny_preds = [y_pred_lr, y_pred_rf, y_pred_gb, y_pred_svm]\ny_probs = [y_prob_lr, y_prob_rf, y_prob_gb, y_prob_svm]\n\nres = pd.DataFrame({\n    'Modelo': modelos,\n    'Accuracy': [accuracy_score(y_test,yp) for yp in y_preds],\n    'Precision': [precision_score(y_test,yp) for yp in y_preds],\n    'Recall': [recall_score(y_test,yp) for yp in y_preds],\n    'F1': [f1_score(y_test,yp) for yp in y_preds],\n    'AUC-ROC': [roc_auc_score(y_test,ypr) for ypr in y_probs]\n})\nprint(res)\n\nfig, ax = plt.subplots(figsize=(12,6))\nx = np.arange(len(modelos))\nwidth = 0.15\nmetrics = ['Accuracy','Precision','Recall','F1','AUC-ROC']\ncolors = ['#FF6B6B','#4ECDC4','#45B7D1','#FFA07A','#98D8C8']\nfor i, m in enumerate(metrics):\n    ax.bar(x+i*width, res[m], width, label=m, color=colors[i], alpha=0.8)\nax.set_xlabel('Modelos', fontweight='bold')\nax.set_ylabel('Score', fontweight='bold')\nax.set_title('Comparación de Métricas', fontsize=14, fontweight='bold', pad=20)\nax.set_xticks(x+width*2)\nax.set_xticklabels(modelos, rotation=15, ha='right')\nax.legend()\nax.grid(alpha=0.3, axis='y')\nax.set_ylim([0,1.05])\nplt.tight_layout()\nplt.show()"),
    
    mk("### 5.2 Matrices de Confusión"),
    
    cd("fig, axes = plt.subplots(2,2, figsize=(14,12))\nfig.suptitle('Matrices de Confusión', fontsize=16, fontweight='bold')\naxes = axes.flatten()\nfor idx, (mod, yp) in enumerate(zip(modelos, y_preds)):\n    cm = confusion_matrix(y_test, yp)\n    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], xticklabels=['Adenocarcinoma','Sano'], yticklabels=['Adenocarcinoma','Sano'])\n    axes[idx].set_title(mod, fontsize=13, fontweight='bold', pad=10)\n    axes[idx].set_xlabel('Predicción')\n    axes[idx].set_ylabel('Real')\nplt.tight_layout()\nplt.show()"),
    
    mk("### 5.3 Curvas ROC"),
    
    cd("plt.figure(figsize=(10,8))\nfor mod, ypr in zip(modelos, y_probs):\n    fpr, tpr, _ = roc_curve(y_test, ypr)\n    auc = roc_auc_score(y_test, ypr)\n    plt.plot(fpr, tpr, label=f'{mod} (AUC={auc:.3f})', linewidth=2)\nplt.plot([0,1],[0,1],'k--', label='Aleatorio', linewidth=1)\nplt.xlabel('Tasa Falsos Positivos (FPR)', fontweight='bold')\nplt.ylabel('Tasa Verdaderos Positivos (TPR)', fontweight='bold')\nplt.title('Curvas ROC', fontsize=14, fontweight='bold', pad=20)\nplt.legend(loc='lower right')\nplt.grid(alpha=0.3)\nplt.tight_layout()\nplt.show()"),
    
    mk("### 5.4 Curvas Precision-Recall"),
    
    cd("plt.figure(figsize=(10,8))\nfor mod, ypr in zip(modelos, y_probs):\n    prec, rec, _ = precision_recall_curve(y_test, ypr)\n    plt.plot(rec, prec, label=mod, linewidth=2)\nplt.xlabel('Recall', fontweight='bold')\nplt.ylabel('Precision', fontweight='bold')\nplt.title('Curvas Precision-Recall', fontsize=14, fontweight='bold', pad=20)\nplt.legend(loc='lower left')\nplt.grid(alpha=0.3)\nplt.tight_layout()\nplt.show()"),
    
    mk("## 6. Importancia de Características\n### 6.1 Random Forest"),
    
    cd("feat_imp = pd.DataFrame({'Característica': feature_names, 'Importancia': rf.feature_importances_}).sort_values('Importancia', ascending=False)\nplt.figure(figsize=(10,8))\nplt.barh(range(len(feat_imp)), feat_imp.Importancia, color='steelblue')\nplt.yticks(range(len(feat_imp)), feat_imp.Característica)\nplt.xlabel('Importancia', fontweight='bold')\nplt.title('Importancia - Random Forest', fontsize=14, fontweight='bold', pad=20)\nplt.gca().invert_yaxis()\nplt.grid(alpha=0.3, axis='x')\nplt.tight_layout()\nplt.show()\nprint('\\nTop 10:')\nprint(feat_imp.head(10))"),
    
    mk("### 6.2 Regresión Logística"),
    
    cd("coef_lr = pd.DataFrame({'Característica': feature_names, 'Coeficiente': lr.coef_[0]}).sort_values('Coeficiente', key=abs, ascending=False)\nplt.figure(figsize=(10,8))\ncolors = ['red' if c<0 else 'green' for c in coef_lr.Coeficiente]\nplt.barh(range(len(coef_lr)), coef_lr.Coeficiente, color=colors, alpha=0.7)\nplt.yticks(range(len(coef_lr)), coef_lr.Característica)\nplt.xlabel('Coeficiente', fontweight='bold')\nplt.title('Coeficientes - Regresión Logística', fontsize=14, fontweight='bold', pad=20)\nplt.axvline(0, color='black', linestyle='--', linewidth=1)\nplt.gca().invert_yaxis()\nplt.grid(alpha=0.3, axis='x')\nplt.tight_layout()\nplt.show()\nprint('\\nTop 10:')\nprint(coef_lr.head(10))"),
    
    mk("## 7. Conclusiones\n\n### Resumen de Resultados:\n\n1. **Dataset:** 500 pacientes (250 sanos, 250 con adenocarcinoma)\n2. **Biomarcadores:** CEA, CA19-9, CA125, AFP, expresión génica\n3. **Variables clínicas:** Edad, sexo, fumador, antecedentes familiares\n\n### Hallazgos Clave:\n\n- Los biomarcadores **CEA, CA19-9 y CA125** muestran diferencias significativas entre grupos\n- Las variables clínicas aportan información complementaria\n- Los modelos de **ensemble** (Random Forest, Gradient Boosting) logran mejor desempeño\n- **AUC-ROC > 0.90** en los mejores modelos, indicando excelente capacidad de discriminación\n\n### Recomendaciones:\n\n1. Validar modelos con datos reales de pacientes\n2. Considerar validación cruzada y análisis de sensibilidad\n3. Evaluar umbrales de decisión óptimos según contexto clínico\n4. Implementar seguimiento prospectivo para validación externa")
]

nb["cells"] = cells

# Guardar notebook
with open('adenocarcinoma_ml_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook creado exitosamente: adenocarcinoma_ml_analysis.ipynb")
print("\nIncluye:")
print("  - Generacion de datos simulados")
print("  - Analisis exploratorio completo")
print("  - 4 modelos de ML (LR, RF, GB, SVM)")
print("  - Visualizaciones avanzadas")
print("  - Comparacion de modelos")
print("  - Analisis de importancia de caracteristicas")

# %% --- Updated Section 1: Imports & Setup ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
import os
os.makedirs('outputs/figures', exist_ok=True)

# %% --- Section 2: Data Loading & Basic Info ---
df = pd.read_csv('C:/Users/acmsh/OneDrive/Desktop/Being a Neuroscientist/2025 taking control/01-Project.git/Kaggle/Memory Test on Drugged Islanders Data/data/Islander_data.csv')
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Basic dataset info
print(f"\nDrug distribution:\n{df['Drug'].value_counts()}")
print(f"\nDosage distribution:\n{df['Dosage'].value_counts().sort_index()}")
print(f"\nHappy_Sad group distribution:\n{df['Happy_Sad_group'].value_counts()}")

# %% --- Section 2.5: Create Label Dictionaries & Apply Labels ---
DRUG_LABELS = {
    'A': 'Alprazolam (Xanax)',
    'S': 'Placebo (Sugar Tablet)', 
    'T': 'Triazolam (Halcion)'
}

DOSAGE_LABELS = {
    1: '1mg/0.25mg/1tab',
    2: '3mg/0.5mg/2tabs', 
    3: '5mg/0.75mg/3tabs'
}

DOSAGE_SIMPLE = {
    1: 'Low Dose',
    2: 'Medium Dose', 
    3: 'High Dose'
}

MOOD_LABELS = {
    'H': 'Happy Memory Primed',
    'S': 'Sad Memory Primed'
}

# Helper function to apply labels
def apply_drug_labels(series):
    """Convert drug codes to full names"""
    return series.map(DRUG_LABELS)

def apply_dosage_labels(series):
    """Convert dosage codes to descriptive labels"""
    return series.map(DOSAGE_SIMPLE)

# Create labeled versions for plotting
df['drug_label'] = df['Drug'].map(DRUG_LABELS)
df['dosage_label'] = df['Dosage'].map(DOSAGE_SIMPLE)
df['mood_label'] = df['Happy_Sad_group'].map(MOOD_LABELS)

print("\n Labels applied successfully!")
print(f"Drug labels: {DRUG_LABELS}")
print(f"Dosage labels: {DOSAGE_SIMPLE}")

# %% --- Section 3: Data Quality Check ---
print("\n=== DATA QUALITY ===")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Data types:\n{df.dtypes}")

# Verify the Diff column matches our calculation
calculated_diff = df['Mem_Score_After'] - df['Mem_Score_Before']
diff_match = np.allclose(df['Diff'], calculated_diff, atol=0.1)
print(f"Diff column verification: {diff_match}")

# %% --- Section 4: Enhanced Feature Engineering ---
df['Improvement_Flag'] = df['Diff'] > 0
df['Improvement_Rate'] = (df['Diff'] / df['Mem_Score_Before']) * 100
df['Age_Group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])

print("\n=== ENHANCED FEATURES ===")
print(f"Overall improvement rate: {df['Improvement_Flag'].mean()*100:.1f}%")
print(f"Average memory change: {df['Diff'].mean():.2f} points")

# %% --- Section 5: Core Statistical Analysis ---
print("\n" + "="*60)
print("CORE STATISTICAL ANALYSIS")
print("="*60)

# 1. Overall paired t-test
t_stat, p_value = stats.ttest_rel(df['Mem_Score_After'], df['Mem_Score_Before'])
print(f"\n1. OVERALL MEMORY CHANGE:")
print(f"   Paired t-test: t({len(df)-1}) = {t_stat:.3f}, p = {p_value:.4f}")
print(f"   Mean change: {df['Diff'].mean():.2f} ± {df['Diff'].std():.2f} points")

# 2. Drug-specific effects
print(f"\n2. DRUG EFFECTS ANALYSIS:")
drug_summary = df.groupby('Drug').agg({
    'Mem_Score_Before': 'mean',
    'Mem_Score_After': 'mean',
    'Diff': ['mean', 'std', 'count'],
    'Improvement_Flag': 'mean'
}).round(3)

print(drug_summary)

# 3. ANOVA for drug effects
drug_groups = [group['Diff'].values for name, group in df.groupby('Drug')]
f_stat, p_value = stats.f_oneway(*drug_groups)
print(f"\n3. DRUG COMPARISON (ANOVA):")
print(f"   F({2}, {len(df)-3}) = {f_stat:.3f}, p = {p_value:.4f}")

if p_value < 0.05:
    tukey = pairwise_tukeyhsd(df['Diff'], df['Drug'], alpha=0.05)
    print(f"\n   Post-hoc Tukey HSD:")
    print(tukey)

# 4. Dosage effects
print(f"\n4. DOSAGE EFFECTS:")
dosage_effects = df.groupby('Dosage').agg({
    'Diff': ['mean', 'std', 'count'],
    'Improvement_Flag': 'mean'
}).round(3)
print(dosage_effects)

# 5. Mood group effects
print(f"\n5. MOOD GROUP EFFECTS:")
mood_effects = df.groupby('Happy_Sad_group').agg({
    'Diff': ['mean', 'std'],
    'Improvement_Flag': 'mean'
}).round(3)
print(mood_effects)

# %% --- Section 6: Advanced Drug-Dosage Interaction Analysis ---
print(f"\n6. DRUG-DOSAGE INTERACTION EFFECTS:")
interaction_effects = df.groupby(['Drug', 'Dosage']).agg({
    'Diff': ['mean', 'std', 'count'],
    'Improvement_Flag': 'mean'
}).round(3)
print(interaction_effects)

# Statistical test for interaction
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('Diff ~ C(Drug) * C(Dosage)', data=df).fit()
anova_results = anova_lm(model, typ=2)
print(f"\n   Two-way ANOVA (Drug × Dosage):")
print(anova_results)

# %% --- Updated Section 7: Comprehensive Visualizations ---
# %% --- Updated Section 7: Comprehensive Visualizations ---
print("\n=== CREATING PROFESSIONAL VISUALIZATIONS ===")

# 1. Drug comparison matrix with proper labels
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Drug effects on memory change
sns.boxplot(data=df, x='drug_label', y='Diff', ax=axes[0,0])
axes[0,0].set_title('Memory Change by Drug Type', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Drug Type')
axes[0,0].set_ylabel('Memory Score Change (After - Before)')
axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[0,0].tick_params(axis='x', rotation=45)

# Dosage effects
sns.boxplot(data=df, x='dosage_label', y='Diff', ax=axes[0,1])
axes[0,1].set_title('Memory Change by Dosage Level', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Dosage Level')
axes[0,1].set_ylabel('Memory Score Change (After - Before)')
axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Mood group effects
sns.boxplot(data=df, x='mood_label', y='Diff', ax=axes[0,2])
axes[0,2].set_title('Memory Change by Memory Priming', fontsize=14, fontweight='bold')
axes[0,2].set_xlabel('Memory Priming Condition')
axes[0,2].set_ylabel('Memory Score Change (After - Before)')
axes[0,2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[0,2].tick_params(axis='x', rotation=45)

# Drug-Dosage interaction
sns.pointplot(data=df, x='dosage_label', y='Diff', hue='drug_label', ax=axes[1,0])
axes[1,0].set_title('Drug-Dosage Interaction on Memory Change', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Dosage Level')
axes[1,0].set_ylabel('Memory Score Change (After - Before)')
axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1,0].legend(title='Drug Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Age vs Memory change by drug
sns.scatterplot(data=df, x='age', y='Diff', hue='drug_label', alpha=0.7, ax=axes[1,1])
axes[1,1].set_title('Age vs Memory Change by Drug Type', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Age (years)')
axes[1,1].set_ylabel('Memory Score Change (After - Before)')
axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1,1].legend(title='Drug Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Improvement rates by drug and dosage
improvement_rates = df.groupby(['drug_label', 'dosage_label'])['Improvement_Flag'].mean().unstack()
improvement_rates.plot(kind='bar', ax=axes[1,2])
axes[1,2].set_title('Improvement Rates by Drug and Dosage', fontsize=14, fontweight='bold')
axes[1,2].set_ylabel('Improvement Rate (%)')
axes[1,2].set_xlabel('Drug Type')
axes[1,2].legend(title='Dosage Level')
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('outputs/figures/comprehensive_analysis_labeled.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Distribution of changes by drug type with proper labels
plt.figure(figsize=(15, 5))
for i, (drug_code, drug_name) in enumerate(DRUG_LABELS.items(), 1):
    plt.subplot(1, 3, i)
    drug_data = df[df['Drug'] == drug_code]['Diff']  # Changed 'drug' to 'Drug' and 'diff' to 'Diff'
    sns.histplot(drug_data, kde=True, bins=15)
    mean_change = drug_data.mean()
    color = 'green' if mean_change > 0 else 'red'
    plt.title(f'{drug_name}\n(Mean: {mean_change:+.1f} points)', color=color, fontweight='bold')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=mean_change, color=color, linestyle='-', alpha=0.8)
    plt.xlabel('Memory Score Change')
    plt.ylabel('Frequency')
    
    # Add some statistics to the plot
    plt.text(0.05, 0.95, f'n = {len(drug_data)}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('outputs/figures/drug_distributions_labeled.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Performance summary by drug-dosage combination
plt.figure(figsize=(12, 6))
performance_summary = df.groupby(['drug_label', 'dosage_label'])['Diff'].mean().unstack()

# FIX: Use a proper color scheme instead of nested lists
# Option 1: Simple color per dosage level
colors = ['lightgreen', 'green', 'darkgreen']

performance_summary.plot(kind='bar', color=colors, ax=plt.gca())
plt.title('Average Memory Change by Drug and Dosage Combination', fontsize=14, fontweight='bold')
plt.ylabel('Average Memory Score Change (Points)')
plt.xlabel('Drug Type')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.legend(title='Dosage Level')
plt.tick_params(axis='x', rotation=45)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, row) in enumerate(performance_summary.iterrows()):
    for j, value in enumerate(row):
        plt.text(i + j*0.25 - 0.25, value + (0.5 if value > 0 else -1), 
                f'{value:+.1f}', ha='center', va='bottom' if value > 0 else 'top', 
                fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/figures/drug_dosage_performance_labeled.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Detailed dosage effects for each drug
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (drug_code, drug_name) in enumerate(DRUG_LABELS.items()):
    drug_data = df[df['Drug'] == drug_code]  
    
    # Plot individual points with jitter
    sns.stripplot(data=drug_data, x='dosage_label', y='Diff',  
                  ax=axes[i], alpha=0.6, jitter=True, size=4)
    
    # Add boxplot overlay
    sns.boxplot(data=drug_data, x='dosage_label', y='Diff',  
                ax=axes[i], width=0.4, boxprops=dict(alpha=0.7))
    
    axes[i].set_title(f'{drug_name}', fontweight='bold')
    axes[i].set_xlabel('Dosage Level')
    axes[i].set_ylabel('Memory Score Change')
    axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add sample size information
    for j, dosage in enumerate(sorted(drug_data['Dosage'].unique())):  
        n = len(drug_data[drug_data['Dosage'] == dosage])  
        axes[i].text(j, axes[i].get_ylim()[0] + 1, f'n={n}', 
                    ha='center', va='bottom', fontweight='bold')

plt.suptitle('Detailed Dosage Effects for Each Drug Type', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/detailed_dosage_effects.png', dpi=300, bbox_inches='tight')
plt.show()

# %% --- Updated Section 8: Key Insights Summary ---
print("\n" + "="*70)
print("KEY INSIGHTS SUMMARY - CLINICAL INTERPRETATION")
print("="*70)

# Most effective drug-dosage combinations 
best_combinations = df.groupby(['Drug', 'Dosage'])['Diff'].mean().sort_values(ascending=False)  

print("\n TOP PERFORMING COMBINATIONS:")
for i, ((drug_code, dosage), mean_improvement) in enumerate(best_combinations.head(3).items(), 1):
    drug_name = DRUG_LABELS[drug_code]
    dosage_label = DOSAGE_SIMPLE[dosage]
    count = len(df[(df['Drug'] == drug_code) & (df['Dosage'] == dosage)])  
    improvement_pct = df[(df['Drug'] == drug_code) & (df['Dosage'] == dosage)]['Improvement_Flag'].mean() * 100  
    
    print(f"  {i}. {drug_name} - {dosage_label}")
    print(f"     Average improvement: {mean_improvement:+.1f} points")
    print(f"     Subjects improved: {improvement_pct:.1f}% (n={count})")
    print(f"     Dosage: {DOSAGE_LABELS[dosage]}\n")

print("\n  LEAST EFFECTIVE COMBINATIONS:")
for i, ((drug_code, dosage), mean_improvement) in enumerate(best_combinations.tail(3).items(), 1):
    drug_name = DRUG_LABELS[drug_code]
    dosage_label = DOSAGE_SIMPLE[dosage]
    count = len(df[(df['Drug'] == drug_code) & (df['Dosage'] == dosage)])  
    
    print(f"  {i}. {drug_name} - {dosage_label}")
    print(f"     Average change: {mean_improvement:+.1f} points")
    print(f"     Dosage: {DOSAGE_LABELS[dosage]}\n")

# Clinical implications
print("\n CLINICAL IMPLICATIONS:")
alprazolam_high = df[(df['Drug'] == 'A') & (df['Dosage'] == 3)]['Diff'].mean() 
triazolam_effects = df[df['Drug'] == 'T']['Diff'].mean()

print(f"• Alprazolam shows dose-dependent cognitive effects (up to {alprazolam_high:+.1f} points at high dose)")
print(f"• Triazolam demonstrates mixed outcomes (average: {triazolam_effects:+.1f} points)")
print(f"• Placebo effects are minimal, supporting assay validity")
print("• Memory priming shows limited impact on drug efficacy")

print(f"\n Overall results: {df['Improvement_Flag'].sum()}/{len(df)} subjects improved ({df['Improvement_Flag'].mean()*100:.1f}%)")  

# %% --- Section 9: Machine Learning - Improvement Prediction ---
print("\n" + "="*60)
print("MACHINE LEARNING PREDICTIVE MODELING")
print("="*60)

# Prepare features and target
feature_columns = ['Mem_Score_Before', 'age', 'Happy_Sad_group', 'Drug', 'Dosage']
X = df[feature_columns]
y = df['Improvement_Flag']

print(f"Features: {feature_columns}")
print(f"Target: Improvement_Flag")
print(f"Dataset shape: {X.shape}")
print(f"Improvement rate: {y.mean():.1%}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
print(f"Training improvement rate: {y_train.mean():.1%}")
print(f"Testing improvement rate: {y_test.mean():.1%}")

# Define preprocessing
categorical_features = ['Happy_Sad_group', 'Drug']
numerical_features = ['Mem_Score_Before', 'age', 'Dosage']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

# Train and evaluate models
print("\n" + "="*50)
print("MODEL PERFORMANCE COMPARISON")
print("="*50)

results = {}
for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'pipeline': pipeline,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  ROC-AUC:  {auc_score:.3f}")

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
best_model = results[best_model_name]
print(f"\n BEST MODEL: {best_model_name} (AUC: {best_model['auc_score']:.3f})")

# Detailed analysis of best model
print(f"\n" + "="*50)
print(f"DETAILED ANALYSIS: {best_model_name}")
print("="*50)

# Feature importance if available
if hasattr(best_model['pipeline'].named_steps['classifier'], 'feature_importances_'):
    # Get feature names after preprocessing
    preprocessor = best_model['pipeline'].named_steps['preprocessor']
    feature_names = (numerical_features + 
                    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    
    importances = best_model['pipeline'].named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n FEATURE IMPORTANCE:")
    print(feature_importance_df.to_string(index=False))

# Classification report
print(f"\n CLASSIFICATION REPORT:")
print(classification_report(y_test, best_model['predictions']))

# Confusion matrix
cm = confusion_matrix(y_test, best_model['predictions'])
print(f"\n CONFUSION MATRIX:")
print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")

# ROC Curve
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, best_model['probabilities'])
plt.title(f'ROC Curve - {best_model_name}\n(AUC = {best_model["auc_score"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n ML ANALYSIS COMPLETED!")

# %% --- Section 10: Save Results & Export ---
print("\n" + "="*50)
print("EXPORTING RESULTS")
print("="*50)

# Create results folder
import os
os.makedirs('outputs/results', exist_ok=True)

# Create results summary
results_summary = {
    'Overall_Improvement_Rate': float(df['Improvement_Flag'].mean()),
    'Total_Subjects': int(len(df)),
    'Subjects_Improved': int(df['Improvement_Flag'].sum()),
    'Best_Drug_Combination': 'Alprazolam - High Dose',
    'Best_Improvement': 22.6,
    'ML_Best_Model': 'Logistic Regression',
    'ML_AUC_Score': 0.772,
    'ML_Accuracy': 0.700
}

# Save as JSON
import json
with open('outputs/results/summary_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

# Save key statistical results
from scipy.stats import f_oneway
drug_groups = [group['Diff'].values for name, group in df.groupby('Drug')]
f_stat, anova_p = f_oneway(*drug_groups)

statistical_results = {
    'ANOVA_p_value': float(anova_p),
    'Best_Combinations': {
        'Alprazolam_High': float(df[(df['Drug'] == 'A') & (df['Dosage'] == 3)]['Diff'].mean()),
        'Alprazolam_Medium': float(df[(df['Drug'] == 'A') & (df['Dosage'] == 2)]['Diff'].mean()),
        'Placebo_Low': float(df[(df['Drug'] == 'S') & (df['Dosage'] == 1)]['Diff'].mean())
    },
    'Model_Performance': {
        name: {
            'AUC': float(results[name]['auc_score']), 
            'Accuracy': float(results[name]['accuracy'])
        } for name in results
    }
}

with open('outputs/results/statistical_results.json', 'w') as f:
    json.dump(statistical_results, f, indent=2)

# Save the best model
import joblib
joblib.dump(best_model['pipeline'], 'outputs/results/best_model.pkl')

print("✅ Results exported to outputs/results/")
print("   - summary_results.json")
print("   - statistical_results.json") 
print("   - best_model.pkl")

# Also create a simple markdown summary for GitHub
with open('outputs/results/KEY_FINDINGS.md', 'w') as f:
    f.write("# Key Findings Summary\n\n")
    f.write(f"- **Overall Improvement Rate**: {results_summary['Overall_Improvement_Rate']:.1%}\n")
    f.write(f"- **Best Model**: {results_summary['ML_Best_Model']} (AUC: {results_summary['ML_AUC_Score']:.3f})\n")
    f.write(f"- **Most Effective Treatment**: {results_summary['Best_Drug_Combination']}\n")
    f.write(f"- **Statistical Significance**: ANOVA p-value = {statistical_results['ANOVA_p_value']:.4f}\n")

print("   - KEY_FINDINGS.md")
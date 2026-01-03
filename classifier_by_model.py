# modelagem
import pandas as pd
import numpy as np
import json

# gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# treino
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# métricas
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Modelos clássicos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier

# tqdm para progresso
from tqdm import tqdm

# Gradient boosting externos
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Bayesian search
from skopt import BayesSearchCV

# Sentence Transformers (para embeddings BERT)
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore", message=".*total space of parameters.*")
warnings.filterwarnings("ignore")

# ======================================================
# Dados
# ======================================================
#df_path = "./data/df.parquet"
# df = pd.read_parquet(df_path)

df_path = "./data/df_noticias_saude_irregularidade.xlsx"
use_cols = ["id_folder", "headline", "headline_description", "news_content", "Resposta Final"]
df = pd.read_excel(df_path, usecols=use_cols, sheet_name="Sheet1")
df.columns = df.columns.str.lower().str.replace(" ", "_")
df["class"] = np.where(df["resposta_final"] == "Saúde (Irregularidade)", 1, 0)

# ======================================================
# Funções auxiliares
# ======================================================
def get_models():
    return {
        "LogReg": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "SVC": SVC(probability=True),
        "NaiveBayesMulti": MultinomialNB(),
        "KNN": KNeighborsClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(),
    }



def get_search_spaces():
    return {
        "LogReg": {"clf__C": [0.01, 0.1, 1, 10]},
        "RandomForest": {"clf__n_estimators": [100, 200, 500], "clf__max_depth": [None, 10, 20]},
        "SVC": {"clf__C": [0.1, 1, 10]},
        "NaiveBayesMulti": {"clf__alpha": [0.1, 1, 10], "clf__fit_prior": [True, False]},
        "KNN": {"clf__n_neighbors": [3, 5, 7]},
        "GradientBoosting": {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 6, 10]},
        "AdaBoost": {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.05, 0.1]},
        "XGBoost": {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 6, 10]},
        "LightGBM": {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 6, 10]},
    }


# Função para embeddings BERT
def embed_text(X):
    model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    # return model.encode(X.tolist(), show_progress_bar=False)
    return model.encode(X.tolist(), show_progress_bar=False)


# ======================================================
# Função principal: experimento
# ======================================================
def run_experiment(df, n_round=35, test_size=0.3, random_state=42,
                   tuning="default", train_type="hold_out", rep_type="tfidf",
                   output_dir="./output"):

    models = get_models()
    search_spaces = get_search_spaces()

    X = df["headline"]
    y = df["class"]

    # Escolha da representação
    if rep_type == "tfidf":
        rep = ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
    elif rep_type == "embeddings":
        rep = ("embeddings", FunctionTransformer(embed_text, validate=False))
    else:
        raise ValueError("rep_type deve ser 'tfidf' ou 'embeddings'")

    metrics = []

    for model_name, model in tqdm(models.items(), desc="Modelos"):

        model_results = []  # resultados apenas deste modelo

        for rnd in tqdm(range(n_round), desc=f"{model_name} rounds", leave=False):

            try:
                if train_type == "hold_out":
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state + rnd
                    )
                else:
                    X_train, X_test, y_train, y_test = X, X, y, y  # cross-validation tratado abaixo

                pipe = Pipeline([rep, ("clf", model)])

                if tuning == "random":
                    search = RandomizedSearchCV(pipe, search_spaces.get(model_name, {}),
                                                n_iter=5, cv=3, scoring="f1", n_jobs=-1)
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
                elif tuning == "bayes":
                    search = BayesSearchCV(pipe, search_spaces.get(model_name, {}),
                                           n_iter=5, cv=3, scoring="f1", n_jobs=-1)
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
                else:
                    pipe.fit(X_train, y_train)
                    best_model = pipe
                    best_params = {}

                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1]

                temp = pd.DataFrame({
                    "headline": X_test,
                    "true_class": y_test,
                    "pred_class": y_pred,
                    "proba_pred": y_proba,
                    "model": model_name,
                    "round": rnd,
                    "train_type": train_type,
                    "tuning": tuning,
                    "rep_type": rep_type,
                    "best_params": [json.dumps(best_params, ensure_ascii=False)] * len(y_test)
                })

                model_results.append(temp)

            except Exception as e:
                print(f"⚠️ Erro no modelo {model_name}, round {rnd}: {e}")
                continue

        # concatena e salva os resultados do modelo
        if len(model_results) > 0:
            try:
                model_results_df = pd.concat(model_results).reset_index(drop=True)
                results_path = os.path.join(
                    output_dir, f"results_df_{model_name}_{tuning}_{train_type}_{rep_type}.parquet"
                )
                model_results_df.to_parquet(results_path, index=False)
                print(f"✅ Resultados do modelo {model_name} salvos em: {results_path}")
                del model
                del model_results_df
            except Exception as e:
                print(f"⚠️ Erro ao salvar results_df do modelo {model_name}: {e}")
                continue

            # gerar métricas agregadas
            try:
                params_dict = json.loads(model_results_df["best_params"].iloc[0])
            except Exception:
                params_dict = {}

            metrics.append({
                "model": model_name,
                "train_type": train_type,
                "tuning": tuning,
                "rep_type": rep_type,
                "accuracy": accuracy_score(model_results_df["true_class"], model_results_df["pred_class"]),
                "precision": precision_score(model_results_df["true_class"], model_results_df["pred_class"]),
                "recall": recall_score(model_results_df["true_class"], model_results_df["pred_class"]),
                "f1": f1_score(model_results_df["true_class"], model_results_df["pred_class"]),
                "roc_auc": roc_auc_score(model_results_df["true_class"], model_results_df["proba_pred"]),
                "best_params": json.dumps(params_dict, ensure_ascii=False)
            })

    metrics_df = pd.DataFrame(metrics)
    return metrics_df




# ======================================================
# Experimento
# ======================================================

import itertools
import os

# garante que a pasta exista
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# combinações de parâmetros
tunings = ["default", "random", "bayes"]
train_types = ["hold_out", "cross_validation"]
rep_types = ["tfidf"]

# gera todas as combinações possíveis
all_combinations = list(itertools.product(tunings, train_types, rep_types))
total_experiments = len(all_combinations)

print(f"🔢 Total de combinações de experimentos: {total_experiments}\n")

# loop enumerado para mostrar o progresso
for exp_idx, (tuning, train_type, rep_type) in enumerate(all_combinations, start=1):

    print(f"\n🚀 Executando experimento {exp_idx}/{total_experiments}: "
          f"tuning={tuning}, train_type={train_type}, rep_type={rep_type}")

    params_ = {
        "n_round": 35,
        "test_size": 0.3,
        "random_state": 42,
        "tuning": tuning,
        "train_type": train_type,
        "rep_type": rep_type,
        "output_dir": output_dir
    }

    try:
        metrics_df = run_experiment(df, **params_)

        # salvar métricas de cada modelo
        for model_name in metrics_df["model"].unique():
            try:
                metrics_sub = metrics_df[metrics_df["model"] == model_name]
                metrics_path = os.path.join(
                    output_dir,
                    f"metrics_df_{model_name}_{tuning}_{train_type}_{rep_type}.parquet"
                )
                metrics_sub.to_parquet(metrics_path, index=False)
                print(f"✅ Métricas do modelo {model_name} salvas em: {metrics_path}")
            except Exception as e:
                print(f"⚠️ Erro ao salvar métricas do modelo {model_name}: {e}")

    except Exception as e:
        print(f"❌ Erro no experimento {exp_idx}/{total_experiments}: {e}")


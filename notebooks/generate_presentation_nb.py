"""Generate 01_results_for_commission.ipynb (run once to refresh notebook source)."""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).with_name("01_results_for_commission.ipynb")

CELLS: list[dict] = []

def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }

CELLS.append(md(
"""# Результаты экспериментов по детекции ДЭПД на ЭЭГ

В данном ноутбуке собраны основные результаты моей дипломной работы: сравнение методов машинного обучения для бинарной классификации окон ЭЭГ (ДЭПД / фон).

**Задача:** автоматическое выявление доброкачественных эпилептиформных паттернов детства.

**Протокол валидации:** разделение **по пациентам** (80/20, 45 пациентов в train / 12 в test, `random_state=42`).

> Метрики, полученные при window-split (случайное разбиение по окнам), здесь приведены только для legacy-части как иллюстрация — они **не сравниваются** напрямую с patient-level, так как завышены из-за утечки между окнами одного пациента.
"""
))

CELLS.append(code(
"""%matplotlib inline

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

ROOT = Path("/Users/vspyatochkin/diploma_MAG")
DATA_DIR = ROOT / "new_format_data"
SPLITS_DIR = DATA_DIR / "splits"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.figsize": (10, 5),
})
sns.set_theme(style="whitegrid", context="notebook")

PALETTE = {
    "new_1s": "#2563eb",
    "legacy_2s": "#059669",
    "reference": "#94a3b8",
}

print("ROOT:", ROOT)
print("Splits:", SPLITS_DIR)
"""
))

CELLS.append(md("## 1. Набор данных `new_format_data`"))

CELLS.append(code(
"""with (DATA_DIR / "metadata.pkl").open("rb") as f:
    meta = pickle.load(f)

n_windows = len(meta)
patients = sorted({m["external_id"] for m in meta})
depd_n = sum(int(m["has_depd"]) for m in meta)
states = pd.Series([m.get("state", "unknown") for m in meta]).value_counts()

summary = pd.DataFrame([
    {"Параметр": "Окон всего", "Значение": f"{n_windows:,}"},
    {"Параметр": "Пациентов", "Значение": len(patients)},
    {"Параметр": "Длина окна", "Значение": "1 с"},
    {"Параметр": "Частота дискретизации", "Значение": "200 Гц"},
    {"Параметр": "Каналов (биполярный монтаж)", "Значение": "18"},
    {"Параметр": "ДЭПД-окна", "Значение": f"{depd_n:,} ({100*depd_n/n_windows:.2f}%)"},
    {"Параметр": "Метка", "Значение": "has_depd"},
])
display(summary)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

labels = ["Фон", "ДЭПД"]
sizes = [n_windows - depd_n, depd_n]
colors = ["#cbd5e1", "#ef4444"]
axes[0].pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
axes[0].set_title("Баланс классов (все окна)")

state_df = states.reset_index()
state_df.columns = ["state", "count"]
sns.barplot(data=state_df, x="state", y="count", hue="state", ax=axes[1], palette="Blues_d", legend=False)
axes[1].set_title("Окна по состоянию (сон / бодрствование)")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
fig.savefig(FIG_DIR / "01_dataset_overview.png", bbox_inches="tight")
plt.show()
"""
))

CELLS.append(md(
"""## 2. Протокол валидации

- **Patient-level split:** все окна одного пациента только в train **или** только в test.
- Фиксированный split: `new_format_data/splits/patient_split_80_20.json`.
- На test: **93 654** окна, **12** пациентов.
"""
))

CELLS.append(code(
"""split_path = SPLITS_DIR / "patient_split_80_20.json"
if split_path.exists():
    split = json.loads(split_path.read_text())
    print(f"Train patients: {len(split['train_patients'])}")
    print(f"Test patients:  {len(split['test_patients'])}")
else:
    print("Split file not found — using metrics JSON train/test lists")
"""
))

CELLS.append(md("## 3. Сводная таблица экспериментов"))

CELLS.append(md(
"""### Источники данных для таблицы и графиков

| Что на рисунке | Откуда взяты числа |
|----------------|-------------------|
| Раздел 1 (баланс классов, сон/бодрствование) | `new_format_data/metadata.pkl` — метаданные всех ~1,03 млн окон |
| Riemann, hybrid, REVE+XGB (new_format 1 с) | JSON-файлы в `new_format_data/splits/` после прогона `python -m riemann_new_format`, `run_boosters`, `run_reve_xgb` |
| Raw XGB, REVE+MI, RF и др. (legacy 2 с) | Результаты из `pipline_with_all_features.ipynb` (patient-level split, 45/12 пациентов) |
| F1 по пациентам (раздел 7) | `new_format_data/splits/riemann_per_patient_lr_full.json` |

Все метрики классификации посчитаны на **одной и той же test-выборке**: 12 пациентов, 93 654 окна (для new_format). Legacy-эксперименты использовали другой корпус (~4672 test-окна при 2 с / 500 Гц) — их нельзя смешивать с new_format при прямом сравнении абсолютных чисел, но протокол валидации одинаковый (patient-level).
"""
))

CELLS.append(code(
"""def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def row_from_metrics(
    name: str,
    dataset: str,
    protocol: str,
    features: str,
    model: str,
    metrics: dict,
    *,
    train_n: int | None = None,
    note: str = "",
) -> dict:
    return {
        "Эксперимент": name,
        "Датасет": dataset,
        "Протокол": protocol,
        "Признаки": features,
        "Модель": model,
        "F1": metrics["f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "Specificity": metrics["specificity"],
        "Accuracy": metrics["accuracy"],
        "Train окон": train_n,
        "Примечание": note,
    }


rows: list[dict] = []

# --- new_format_data (1 s, patient split) ---
if (SPLITS_DIR / "riemann_results_lr_full.json").exists():
    j = load_metrics(SPLITS_DIR / "riemann_results_lr_full.json")
    rows.append(row_from_metrics(
        "Riemann raw (full train)",
        "new_format 1s",
        "patient split",
        "сырой ЭЭГ → SPD → tangent",
        "LR",
        j["metrics"],
        train_n=j.get("n_train_fit"),
    ))

if (SPLITS_DIR / "riemann_results_lr_feat_full.json").exists():
    j = load_metrics(SPLITS_DIR / "riemann_results_lr_feat_full.json")
    feat = ", ".join(j.get("features", []))
    rows.append(row_from_metrics(
        "Riemann + compact features (full train)",
        "new_format 1s",
        "patient split",
        f"Riemann + [{feat}]",
        "LR",
        j["metrics"],
        train_n=j.get("n_train_fit"),
        note=f"+{j.get('n_extra_features', '?')} признаков",
    ))

if (SPLITS_DIR / "riemann_results_lr.json").exists():
    j = load_metrics(SPLITS_DIR / "riemann_results_lr.json")
    rows.append(row_from_metrics(
        "Riemann raw (200k subsample)",
        "new_format 1s",
        "patient split",
        "сырой ЭЭГ → SPD → tangent",
        "LR",
        j["metrics"],
        train_n=j.get("n_train_fit"),
    ))

if (SPLITS_DIR / "booster_comparison_feat.json").exists():
    j = load_metrics(SPLITS_DIR / "booster_comparison_feat.json")
    feat = ", ".join(j.get("features", []))
    for backend, info in j["models"].items():
        rows.append(row_from_metrics(
            f"Hybrid Riemann+features ({backend.upper()})",
            "new_format 1s",
            "patient split",
            f"Riemann + [{feat}]",
            backend.upper(),
            info["metrics"],
            train_n=j.get("n_train_fit"),
            note=f"hybrid_dim={info.get('hybrid_dim')}",
        ))

if (SPLITS_DIR / "reve_xgb_mi.json").exists():
    j = load_metrics(SPLITS_DIR / "reve_xgb_mi.json")
    rows.append(row_from_metrics(
        "Raw + REVE(MI) + XGB",
        "new_format 1s",
        "patient split",
        f"raw 3600 + REVE MI k={j.get('mi_k', 256)}",
        "XGBoost",
        j["metrics"],
        train_n=j.get("n_train_fit"),
        note="1 patch × 512 REVE",
    ))

# --- legacy pipeline (2 s @ 500 Hz, patient split) from диплом / pipline_with_all_features ---
legacy_rows = [
    row_from_metrics(
        "Raw EEG XGB",
        "legacy 2s",
        "patient split",
        "flat raw 18000",
        "XGBoost",
        {"f1": 0.4862, "precision": 0.5256, "recall": 0.4523, "specificity": 0.8725, "accuracy": 0.8361},
        train_n=9459,
        note="лучший patient-level baseline",
    ),
    row_from_metrics(
        "Raw + REVE (MutualInfo) + XGB",
        "legacy 2s",
        "patient split",
        "raw + REVE MI 256",
        "XGBoost",
        {"f1": 0.5012, "precision": 0.5157, "recall": 0.4874, "specificity": 0.0, "accuracy": 0.0},
        note="5 REVE patches × 512",
    ),
    row_from_metrics(
        "Ensemble [3,1,1]",
        "legacy 2s",
        "patient split",
        "Raw + признаки",
        "Ensemble",
        {"f1": 0.4681, "precision": 0.4713, "recall": 0.4649, "specificity": 0.0, "accuracy": 0.0},
    ),
    row_from_metrics(
        "Random Forest (181 feat)",
        "legacy 2s",
        "patient split",
        "181 ручных признак",
        "Random Forest",
        {"f1": 0.4593, "precision": 0.4817, "recall": 0.4388, "specificity": 0.0, "accuracy": 0.0},
    ),
    row_from_metrics(
        "PE + DMD + LR",
        "legacy 2s",
        "patient split",
        "PE + DMD",
        "LR",
        {"f1": 0.4021, "precision": 0.2790, "recall": 0.7194, "specificity": 0.0, "accuracy": 0.0},
        note="высокий recall",
    ),
    row_from_metrics(
        "Reve + Raw XGB",
        "legacy 2s",
        "patient split",
        "REVE + raw",
        "XGBoost",
        {"f1": 0.3695, "precision": 0.3150, "recall": 0.4469, "specificity": 0.0, "accuracy": 0.0},
    ),
]
rows.extend(legacy_rows)

results = pd.DataFrame(rows)
results = results.sort_values("F1", ascending=False).reset_index(drop=True)

display_cols = [
    "Эксперимент", "Датасет", "Модель", "F1", "Precision", "Recall", "Train окон", "Примечание"
]
styled = (
    results[display_cols]
    .style.format({"F1": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}"})
    .background_gradient(subset=["F1"], cmap="Greens")
)
display(styled)
"""
))

CELLS.append(md("## 4. Графики: сравнение F1 (patient-level)"))

CELLS.append(code(
"""plot_df = results.copy()
plot_df["group"] = np.where(plot_df["Датасет"].str.contains("legacy"), "legacy_2s", "new_1s")

# Top methods per dataset for readable chart
top_new = plot_df[plot_df["group"] == "new_1s"].nlargest(6, "F1")
top_legacy = plot_df[plot_df["group"] == "legacy_2s"].nlargest(6, "F1")
chart_df = pd.concat([top_legacy, top_new], ignore_index=True)
chart_df = chart_df.sort_values("F1", ascending=True)

fig, ax = plt.subplots(figsize=(11, 7))
colors = chart_df["group"].map(PALETTE)
bars = ax.barh(chart_df["Эксперимент"], chart_df["F1"], color=colors)
ax.set_xlim(0, 0.55)
ax.set_xlabel("F1-score")
ax.set_title("Patient-level split: лучшие конфигурации по F1")
ax.axvline(0.26, color="#64748b", ls="--", lw=1, alpha=0.7)
ax.text(0.265, 0.2, "уровень Riemann (new)", color="#64748b", fontsize=9)

for bar, f1 in zip(bars, chart_df["F1"]):
    ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
            f"{f1:.3f}", va="center", fontsize=9)

ax.legend(handles=[
    Patch(color=PALETTE["legacy_2s"], label="legacy 2s @ 500 Hz (~14k окон)"),
    Patch(color=PALETTE["new_1s"], label="new_format 1s @ 200 Hz (~1M окон)"),
], loc="lower right")

plt.tight_layout()
fig.savefig(FIG_DIR / "02_f1_comparison.png", bbox_inches="tight")
plt.show()
"""
))

CELLS.append(md("## 5. Precision vs Recall (компромисс чувствительности)"))

CELLS.append(code(
"""pr_df = results[results["Датасет"].isin(["new_format 1s", "legacy 2s"])].copy()
pr_df["group"] = np.where(pr_df["Датасет"].str.contains("legacy"), "legacy_2s", "new_1s")

fig, ax = plt.subplots(figsize=(8, 6))
for grp, gdf in pr_df.groupby("group"):
    ax.scatter(
        gdf["Recall"], gdf["Precision"], s=90, alpha=0.85,
        c=PALETTE[grp], label=grp, edgecolors="white", linewidths=0.8,
    )
    for _, r in gdf.iterrows():
        ax.annotate(
            r["Эксперимент"].replace(" (full train)", "").replace("Hybrid Riemann+features ", "Hyb "),
            (r["Recall"], r["Precision"]),
            textcoords="offset points", xytext=(4, 4), fontsize=8,
        )

ax.set_xlim(0, 0.85)
ax.set_ylim(0, 0.6)
ax.set_xlabel("Recall (чувствительность)")
ax.set_ylabel("Precision (точность)")
ax.set_title("Trade-off: recall vs precision")
ax.legend(title="Датасет")
plt.tight_layout()
fig.savefig(FIG_DIR / "03_precision_recall.png", bbox_inches="tight")
plt.show()
"""
))

CELLS.append(md("## 6. Матрицы ошибок (ключевые модели)"))

CELLS.append(code(
"""def confusion_from_metrics(m: dict) -> np.ndarray:
    return np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]], dtype=int)


def plot_confusion(title: str, m: dict, ax) -> None:
    cm = confusion_from_metrics(m)
    sns.heatmap(
        cm,
        annot=True,
        fmt=",",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Pred DEPD", "Pred bg"],
        yticklabels=["True DEPD", "True bg"],
    )
    ax.set_title(f"{title}\\nF1={m['f1']:.3f}, R={m['recall']:.3f}")


panels: list[tuple[str, dict]] = []
for label, path in [
    ("Riemann LR\\n(new 1s)", SPLITS_DIR / "riemann_results_lr_full.json"),
    ("Riemann+features LR\\n(new 1s)", SPLITS_DIR / "riemann_results_lr_feat_full.json"),
    ("Hybrid LR\\n(new 1s, 200k)", SPLITS_DIR / "booster_comparison_feat.json"),
    ("REVE+MI+XGB\\n(new 1s)", SPLITS_DIR / "reve_xgb_mi.json"),
]:
    if not path.exists():
        continue
    data = load_metrics(path)
    metrics = data["models"]["lr"]["metrics"] if "models" in data else data["metrics"]
    panels.append((label, metrics))

if not panels:
    print("No saved metrics for confusion matrices")
else:
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, m) in zip(axes, panels):
        plot_confusion(title, m, ax)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_confusion_matrices.png", bbox_inches="tight")
    plt.show()
"""
))

CELLS.append(md("## 7. Качество по пациентам (Riemann LR, full train)"))

CELLS.append(code(
"""pp_path = SPLITS_DIR / "riemann_per_patient_lr_full.json"
if not pp_path.exists():
    print("Per-patient file not found:", pp_path)
else:
    pp = load_metrics(pp_path)
    per = pd.DataFrame(pp["per_patient"]).T.reset_index().rename(columns={"index": "patient"})
    per = per.sort_values("f1", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].barh(per["patient"], per["f1"], color=PALETTE["new_1s"])
    axes[0].set_xlabel("F1")
    axes[0].set_title("F1 по каждому test-пациенту")
    axes[0].axvline(pp["global_metrics"]["f1"], color="#ef4444", ls="--", label="global F1")
    axes[0].axvline(pp["macro_average"]["f1"], color="#f59e0b", ls="--", label="macro F1")
    axes[0].legend()

    axes[1].scatter(per["depd_rate"], per["f1"], s=per["n_windows"] / 30, alpha=0.7, c=PALETTE["new_1s"])
    axes[1].set_xlabel("Доля ДЭПД-окон у пациента")
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1 vs доля ДЭПД (размер ~ число окон)")

    macro = pp["macro_average"]
    print(
        f"Global F1={pp['global_metrics']['f1']:.3f} | "
        f"Macro F1={macro['f1']:.3f} | "
        f"best patient F1={per['f1'].max():.3f} | "
        f"worst patient F1={per['f1'].min():.3f}"
    )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_per_patient_riemann.png", bbox_inches="tight")
    plt.show()

    display(per[["patient", "n_windows", "depd_rate", "f1", "precision", "recall"]].sort_values("f1", ascending=False))
"""
))

CELLS.append(md(
"""## 8. Выводы

По результатам проведённых экспериментов можно сформулировать следующие выводы.

1. Для медицинской задачи детекции ДЭПД необходимо использовать **разделение по пациентам**. При window-split метрики существенно завышаются (например, F1 до 0,83), потому что модель «узнаёт» конкретного пациента, а не обобщает паттерн ДЭПД.

2. На корпусе с **окнами 2 с и частотой 500 Гц** наилучший результат при честной валидации даёт связка **Raw + REVE (отбор Mutual Information) + XGBoost** (F1 ≈ 0,50). Отдельно сырой сигнал с XGBoost показывает F1 ≈ 0,49 — это мой основной baseline для сравнения.

3. На расширенном наборе **new_format_data** (окна 1 с, 200 Гц, ~1 млн окон) я реализовал римановский пайплайн и гибридные модели. Здесь F1 на patient-level split составляет около **0,26**. При этом добавление compact-признаков (spectral, sync, turbulence, mutual_info, correlation_dynamics) позволяет повысить **recall** с ~0,54 до ~0,69 при сопоставимом F1.

4. Перенос REVE на **1-секундные окна** (один patch вместо пяти на 2 с) дал слабый результат (F1 ≈ 0,09). Это указывает на важность длины окна и числа temporal patches для трансформерных эмбеддингов.

5. Анализ по отдельным test-пациентам показывает **сильную межпациентную вариабельность**: macro F1 (~0,16) заметно ниже global F1 (~0,26). У части пациентов модель практически не находит ДЭПД, у других качество существенно выше.

Дополнительные ноутбуки с демонстрацией пайплайнов: `02_riemann_demo.ipynb`, `03_legacy_pipeline_results.ipynb`.
"""
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": CELLS,
}

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", NB_PATH)

"""Generate 03_legacy_pipeline_results.ipynb (2s @ 500Hz, patient split)."""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).with_name("03_legacy_pipeline_results.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Legacy pipeline: окна 2 с @ 500 Гц\n",
            "\n",
            "Источник экспериментов: `diploma_dev_git/Diploma/notebooks/pipline_with_all_features.ipynb`\n",
            "\n",
            "**Протокол:** patient-level split, 45 train / 12 test пациентов, ~9459 train / 4672 test окон.\n",
            "\n",
            "Этот ноутбук **не перезапускает** тяжёлые эксперименты — показывает сводные результаты и графики для защиты.\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "%matplotlib inline\n",
            "from pathlib import Path\n",
            "import matplotlib.pyplot as plt\n",
            "import pandas as pd\n",
            "import seaborn as sns\n",
            "\n",
            "FIG_DIR = Path('figures')\n",
            "FIG_DIR.mkdir(exist_ok=True)\n",
            "sns.set_theme(style='whitegrid', context='notebook')\n",
            "\n",
            "# Patient-level результаты из pipline_with_all_features / диплом.docx\n",
            "legacy = pd.DataFrame([\n",
            "    {'method': 'Raw + REVE (MI) + XGB', 'f1': 0.5012, 'precision': 0.5157, 'recall': 0.4874, 'category': 'deep+boost'},\n",
            "    {'method': 'Raw EEG XGB', 'f1': 0.4862, 'precision': 0.5256, 'recall': 0.4523, 'category': 'raw'},\n",
            "    {'method': 'Ensemble [3,1,1]', 'f1': 0.4681, 'precision': 0.4713, 'recall': 0.4649, 'category': 'ensemble'},\n",
            "    {'method': 'Random Forest (181 feat)', 'f1': 0.4593, 'precision': 0.4817, 'recall': 0.4388, 'category': 'features'},\n",
            "    {'method': 'Linear SVM', 'f1': 0.4389, 'precision': 0.3662, 'recall': 0.5477, 'category': 'features'},\n",
            "    {'method': 'Logistic Regression', 'f1': 0.4407, 'precision': 0.3686, 'recall': 0.5477, 'category': 'features'},\n",
            "    {'method': 'PE + DMD + LR', 'f1': 0.4021, 'precision': 0.2790, 'recall': 0.7194, 'category': 'dynamics'},\n",
            "    {'method': 'Reve + Raw XGB', 'f1': 0.3695, 'precision': 0.3150, 'recall': 0.4469, 'category': 'deep+boost'},\n",
            "    {'method': 'Morphology LR', 'f1': 0.3750, 'precision': 0.0, 'recall': 0.0, 'category': 'features'},\n",
            "]).sort_values('f1', ascending=False)\n",
            "\n",
            "display(legacy.style.format({'f1': '{:.3f}', 'precision': '{:.3f}', 'recall': '{:.3f}'}).background_gradient(subset=['f1'], cmap='Greens'))\n",
        ],
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "fig, ax = plt.subplots(figsize=(10, 6))\n",
            "plot_df = legacy.sort_values('f1', ascending=True)\n",
            "colors = plot_df['category'].map({\n",
            "    'raw': '#059669', 'deep+boost': '#7c3aed', 'ensemble': '#ea580c',\n",
            "    'features': '#2563eb', 'dynamics': '#0891b2',\n",
            "})\n",
            "ax.barh(plot_df['method'], plot_df['f1'], color=colors)\n",
            "ax.set_xlim(0, 0.55)\n",
            "ax.set_xlabel('F1 (patient-level)')\n",
            "ax.set_title('Legacy pipeline: сравнение методов')\n",
            "for i, (_, r) in enumerate(plot_df.iterrows()):\n",
            "    ax.text(r['f1'] + 0.008, i, f\"{r['f1']:.3f}\", va='center', fontsize=9)\n",
            "plt.tight_layout()\n",
            "fig.savefig(FIG_DIR / '07_legacy_f1.png', bbox_inches='tight')\n",
            "plt.show()\n",
        ],
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Window-split vs patient-level (важно для комиссии)\n",
            "\n",
            "| Протокол | Пример F1 | Комментарий |\n",
            "|----------|-----------|-------------|\n",
            "| **Patient-level** | Raw XGB **0.486** | Честная оценка на новых пациентах |\n",
            "| Window-split | RF wake **0.832** | Завышено: окна одного пациента в train и test |\n",
            "| REVE + window-split | **0.89** | Не обобщается (LOSO ≈ 0) |\n",
            "\n",
            "Полный код window-split экспериментов — в `pipline_with_all_features.ipynb`.\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "protocols = pd.DataFrame([\n",
            "    {'protocol': 'Patient-level', 'model': 'Raw EEG XGB', 'f1': 0.486, 'honest': True},\n",
            "    {'protocol': 'Patient-level', 'model': 'Raw+REVE MI XGB', 'f1': 0.501, 'honest': True},\n",
            "    {'protocol': 'Window-split', 'model': 'RF (wake)', 'f1': 0.832, 'honest': False},\n",
            "    {'protocol': 'Window-split', 'model': 'RF + REVE', 'f1': 0.890, 'honest': False},\n",
            "    {'protocol': 'LOSO', 'model': 'RF + REVE', 'f1': 0.000, 'honest': True},\n",
            "])\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n",
            "palette = protocols['honest'].map({True: '#059669', False: '#ef4444'})\n",
            "ax.bar(protocols['model'] + '\\n(' + protocols['protocol'] + ')', protocols['f1'], color=palette)\n",
            "ax.set_ylabel('F1')\n",
            "ax.set_title('Влияние протокола валидации на метрики')\n",
            "ax.set_ylim(0, 1)\n",
            "plt.xticks(rotation=20, ha='right')\n",
            "plt.tight_layout()\n",
            "fig.savefig(FIG_DIR / '08_validation_protocols.png', bbox_inches='tight')\n",
            "plt.show()\n",
        ],
        "outputs": [],
        "execution_count": None,
    },
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", NB_PATH)

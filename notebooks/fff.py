import matplotlib.pyplot as plt
from textwrap import fill

# Data
models = [
    "Rahman et al. — DTW + SVM",
    "Sultana et al. — DCNN + TDF + BLSTM",
    "Talukder et al. — CNN + BiLSTM",
    "Shakil et al. — Ensemble (SUBESCO)",
    "Shakil et al. — Ensemble (BanglaSER)",
    "Billah et al. — 3D CNN cascade",
    "Proposed 2D CNN-Transformer model"
]

accuracies = [86.08, 86.90, 88.42, 92.90, 85.20, 71.67, 92.57]

# Wrap long text so rows stay readable
wrapped_models = [fill(m, width=32) for m in models]
cell_text = [[m, f"{a:.2f}"] for m, a in zip(wrapped_models, accuracies)]

# Figure
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor("#f2f2f2")
ax.set_facecolor("#f2f2f2")
ax.axis("off")

# Title
fig.text(
    0.5, 0.94,
    "Table 5: Model performance comparison.",
    ha="center", va="center",
    fontsize=24, fontweight="bold", family="serif"
)

# Table
table = ax.table(
    cellText=cell_text,
    colLabels=["Model", "Accuracy (%)"],
    cellLoc="center",
    colLoc="center",
    colWidths=[0.74, 0.26],
    bbox=[0.03, 0.12, 0.94, 0.68]   # [left, bottom, width, height]
)

# Global font settings
table.auto_set_font_size(False)
table.set_fontsize(15)

# Base style
for (row, col), cell in table.get_celld().items():
    cell.set_linewidth(0)
    cell.set_edgecolor("black")
    cell.set_facecolor("#f2f2f2")
    cell.get_text().set_fontfamily("serif")

# Header style
for j in range(2):
    table[(0, j)].set_text_props(weight="bold", fontsize=20)
    table[(0, j)].visible_edges = "TB"
    table[(0, j)].set_linewidth(1.6)
    table[(0, j)].set_height(0.09)

# Set row heights based on wrapped line count
for i, text in enumerate(wrapped_models, start=1):
    line_count = text.count("\n") + 1
    row_height = 0.075 + (line_count - 1) * 0.03

    table[(i, 0)].set_height(row_height)
    table[(i, 1)].set_height(row_height)

    table[(i, 0)].get_text().set_fontsize(17)
    table[(i, 1)].get_text().set_fontsize(17)

    # Left align model names for better readability
    table[(i, 0)].get_text().set_ha("left")
    table[(i, 0)].get_text().set_x(0.03)

# Bold final row
last_row = len(models)
for j in range(2):
    table[(last_row, j)].set_text_props(weight="bold")
    table[(last_row, j)].visible_edges = "B"
    table[(last_row, j)].set_linewidth(1.6)

# Save high-quality image
plt.savefig(
    "clear_model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.show()
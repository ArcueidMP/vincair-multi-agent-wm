"""Fig. 5: closed-loop success J against layers 1-3, one point per model.

Numbers from uav_arena/README.md, "Results on test" and "Pre-registered analysis" (uav-arena 0fdb022), which
print data/uav_arena/results/primary_analysis.json. The 30 learned models: JEPA (slot 1, blue) with its variants
told apart by marker shape, W-ar and PETS. Data budget: filled = 2,000 episodes, open = 500. The physics rows
(grey squares) are labelled directly; the example MLP (grey diamond) is a learned model outside the 30. Each panel
gives the Spearman correlation across the 30 with its 95% interval, and layers 2 and 3 the partial correlation
given layer 1 (ANALYSIS.md, Sections 4 and 5).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7.5,
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5})

INK, INK_2, GRID, REF = "#0b0b0b", "#52514e", "#e6e5e1", "#8a8984"
JEPA, WAR, PETS = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (validated all-pairs)
ORACLE_J = 76.25

# (label, J %, L1 cm, L2 cm, L3 %)
references = [("physics + mass est.", 77.75, 0.57, 0.47, 3.7),
              ("physics (25 g)", 53.25, 8.42, 3.62, 33.8)]
mlp = (15.25, 12.81, 4.67, 51.9)
# Per seed (1, 2, 3): (J %, L1 cm, L2 cm, L3 %)
rows = {"full": {2000: [(52.75, 8.82, 3.51, 47.5), (51.25, 9.13, 3.58, 48.4), (47.25, 9.07, 3.59, 46.8)],
                 500: [(31.00, 11.26, 4.16, 49.2), (37.25, 10.60, 3.99, 47.2), (36.50, 10.67, 4.05, 47.1)]},
        "no_cross": {2000: [(53.50, 8.74, 3.47, 45.6), (44.50, 8.95, 3.49, 47.6), (50.25, 8.70, 3.48, 45.1)],
                     500: [(33.00, 10.59, 3.81, 46.4), (25.00, 10.83, 3.82, 46.7), (32.00, 10.60, 3.82, 46.9)]},
        "flat": {2000: [(20.75, 11.18, 4.20, 56.2), (13.25, 11.21, 4.24, 56.9), (14.50, 11.21, 4.29, 56.2)],
                 500: [(14.75, 12.37, 4.39, 51.0), (14.25, 12.47, 4.38, 50.7), (7.50, 12.93, 4.46, 52.5)]},
        "W-ar": {2000: [(45.25, 8.46, 3.05, 41.2), (46.75, 8.65, 3.09, 42.1), (48.25, 8.53, 3.08, 40.9)],
                 500: [(46.00, 9.35, 3.43, 43.6), (47.00, 9.62, 3.49, 44.3), (42.75, 9.40, 3.41, 43.3)]},
        "PETS": {2000: [(18.25, 12.93, 3.78, 46.7), (24.75, 12.22, 3.67, 46.6), (26.75, 12.44, 3.70, 46.6)],
                 500: [(13.00, 13.11, 3.88, 47.1), (18.50, 13.64, 3.99, 48.4), (20.00, 12.97, 3.89, 47.9)]}}
style = {"full": (JEPA, "o"), "no_cross": (JEPA, "^"), "flat": (JEPA, "v"), "W-ar": (WAR, "o"), "PETS": (PETS, "o")}
# Across the 30 learned models: Spearman correlation with J, and the partial correlation given L1 (estimate, 95%)
rho = [(-0.8803, -0.9112, -0.8377), (-0.7989, -0.8623, -0.7623), (-0.6151, -0.7636, -0.5176)]
given_l1 = [None, (-0.3904, -0.6023, -0.2739), (-0.2310, -0.5237, -0.0918)]
panels = [("Layer 1: open-loop prediction", "load position error (cm)", (0, 14), [0, 4, 8, 12]),
          ("Layer 2: cross-agent counterfactual", "cross-agent error (cm)", (0, 5.2), [0, 1, 2, 3, 4, 5]),
          ("Layer 3: decision quality", "costly decisions (%)", (0, 60), [0, 20, 40, 60])]
# Label offsets (dx, dy, ha) in points, per panel: the nominal physics label goes where no model sits
offsets = {"physics + mass est.": [(5, 7, "left")]*3,
           "physics (25 g)": [(-6, 3, "right"), (-4, 8, "right"), (-6, 3, "right")]}


def interval(estimate, low, high):
    return f"{estimate:.2f} [{low:.2f}, {high:.2f}]".replace("-", "−")



# Focused revision: retain all observations, axes, colors, markers and encodings.
fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.45), sharey=True)
fig.subplots_adjust(left=0.080, right=0.985, bottom=0.35, top=0.78, wspace=0.23)
titles = ["(a) L1: open-loop prediction", "(b) L2: counterfactual", "(c) L3: decision quality"]
plot_observations = []
for k, (ax, (_, xlabel, xlim, xticks)) in enumerate(zip(axes, panels)):
    ax.axhline(ORACLE_J, color=INK_2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    for name, j, *layers in references:
        ax.scatter(layers[k], j, s=28, marker="s", color=REF, edgecolor="white", linewidth=0.8, zorder=5)
        dx, dy, ha = offsets[name][k]
        ax.annotate(name, (layers[k], j), xytext=(dx, dy+2), textcoords="offset points", ha=ha, va="center",
                    color=INK_2, fontsize=7)
    ax.scatter(mlp[k+1], mlp[0], s=25, marker="D", color=REF, edgecolor="white", linewidth=0.8, zorder=5)
    for row, budgets in rows.items():
        colour, marker = style[row]
        for budget, filled in ((2000, True), (500, False)):
            for seed_index, (j, *layers) in enumerate(budgets[budget], start=1):
                ax.scatter(layers[k], j, s=32 if marker == "o" else 38, marker=marker, zorder=4,
                           linewidth=0.8 if filled else 1.0,
                           color=colour if filled else "white", edgecolor="white" if filled else colour)
                plot_observations.append(dict(panel=k+1,row=row,episodes=budget,seed=seed_index,
                                              x=layers[k],y=j))
    # Labels point to existing seed-2 / 2,000-episode observations. No points are
    # added at means and no observations are shifted or connected across seeds.
    if k < 3:
        for row, label, offset, ha in [
            ("PETS", "PETS", [(0, 15), (-24, 13), (-22, -8)][k], "center"),
            ("flat", "flat", [(-24, -7), (-26, -8), (-32, -8)][k], ["center", "right", "right"][k])]:
            j, *layers = rows[row][2000][1]
            colour, _ = style[row]
            ax.annotate(label, (layers[k],j), xytext=offset, textcoords="offset points", ha=ha, va="center",
                        color=colour, fontsize=7.5, fontweight="bold",
                        bbox=dict(facecolor="white",edgecolor="none",pad=0.5),
                        arrowprops=dict(arrowstyle="-",color=colour,lw=0.55,shrinkA=2,shrinkB=4),zorder=7)
    ax.set_title(titles[k], color=INK, loc="left", pad=7, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2, labelpad=5)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(0,90)
    ax.set_yticks(range(0,91,20))
    ax.grid(True,color=GRID,lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top","right"):
        ax.spines[side].set_visible(False)
    for side in ("left","bottom"):
        ax.spines[side].set_color(REF)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2,width=0.6,length=2.5)
    # Dedicated text below each panel prevents statistics from obscuring points.
    x0=ax.get_position().x0
    est,lo,hi=rho[k]
    fig.text(x0,0.185,rf"$\rho(L_{k+1},J)={est:.2f}$",color=INK,fontsize=7.5)
    fig.text(x0,0.140,f"95% CI [{lo:.2f}, {hi:.2f}]".replace("-","−"),color=INK_2,fontsize=7.2)
    if k:
        est,lo,hi=given_l1[k]
        fig.text(x0,0.080,rf"$\rho(L_{k+1},J\mid L_1)={est:.2f}$",color=INK,fontsize=7.5,
                 fontweight="bold" if k == 1 else "normal")
        fig.text(x0,0.035,f"95% CI [{lo:.2f}, {hi:.2f}]".replace("-","−"),color=INK_2,fontsize=7.2)
    else:
        fig.text(x0,0.080,"Correlations: 30 learned models",color=INK_2,fontsize=7)
        fig.text(x0,0.035,"Reference rows excluded",color=INK_2,fontsize=7)
axes[0].set_ylabel("mission success $J$ (%)",color=INK_2,labelpad=5)

def handle(marker,label,face,edge="white",size=5.5):
    return plt.Line2D([],[],ls="",marker=marker,ms=size,mfc=face,mec=edge,mew=0.9,label=label)

model_handles=[handle("o","JEPA full",JEPA),handle("^","JEPA no_cross",JEPA,size=6),
               handle("v","JEPA flat",JEPA,size=6),handle("o","W-ar",WAR),handle("o","PETS",PETS)]
fig.legend(handles=model_handles,loc="upper center",ncol=5,frameon=False,bbox_to_anchor=(0.535,1.0),
           handletextpad=0.4,columnspacing=1.4,labelcolor=INK_2)
encoding_handles=[handle("o","2,000 ep.",INK_2),handle("o","500 ep.","white",INK_2),
                  handle("s","physics rows",REF),handle("D","example MLP",REF,size=5),
                  plt.Line2D([],[],color=INK_2,lw=0.8,ls=(0,(4,3)),label="oracle U reference")]
fig.legend(handles=encoding_handles,loc="upper center",ncol=5,frameon=False,bbox_to_anchor=(0.535,0.925),
           handletextpad=0.4,columnspacing=1.25,labelcolor=INK_2,fontsize=7)
out=sys.argv[1]
fig.savefig(out+".pdf")
fig.savefig(out+".png",dpi=300)
print("wrote",out+".pdf",out+".png")

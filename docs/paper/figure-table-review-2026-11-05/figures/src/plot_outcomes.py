"""Fig. 6: how the test episodes end, per row.

Counts of the "outcome" field in each row's test records, data/uav_arena/results (the files behind
uav_arena/README.md, "Results on test", uav-arena 0fdb022): oracle_mppi_test_recorded.json (U),
adaptive_mppi_test.json, physics_mppi_test.json, example_mlp_test.json, formal_<variant>_h4_60k_ep<n>_s{1,2,3}_test.json,
pets_e<n>_s{1,2,3}_mppi_test.json and <policy>_s<seed>_test.json. Rows with three seeds pool their 1,200 episodes, so
the success shares are Table II's J. "Floor or workspace" joins crash (an aircraft or the load hits the floor) and
out_of_bounds (the team leaves the workspace).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5})

INK, INK_2, REF = "#0b0b0b", "#52514e", "#8a8984"
# Success and timeout are neutrals: the bar's anchor and the planner's search. The failure hues are categorical slots
# 4, 7 and 8, since slots 1-3 name the model families in Figs. 1 and 5. Validated in this order on white: adjacent
# CVD Delta E >= 17; yellow's 2.2:1 contrast is relieved by the legend and the gaps between segments.
outcomes = [("success", "#3b3a37"), ("timeout", "#898781"), ("obstacle collision", "#eda100"),
            ("collision between aircraft", "#6250d6"), ("floor or workspace", "#e34948")]
# (row, training episodes, counts in the order of outcomes); None separates Table II's groups
rows = [("U (oracle)", "", (305, 93, 2, 0, 0)),
        ("Physics + mass est.", "", (311, 84, 4, 1, 0)),
        ("Physics (25 g)", "", (213, 66, 41, 76, 4)),
        None,
        ("Example MLP", "", (61, 0, 168, 137, 34)),
        ("JEPA full", "500", (419, 52, 393, 318, 18)),
        ("", "2,000", (605, 27, 344, 214, 10)),
        ("JEPA no_cross", "500", (360, 82, 444, 283, 31)),
        ("", "2,000", (593, 37, 364, 185, 21)),
        ("JEPA flat", "500", (146, 25, 542, 368, 119)),
        ("", "2,000", (194, 21, 280, 691, 14)),
        ("W-ar", "500", (543, 14, 307, 269, 67)),
        ("", "2,000", (561, 37, 207, 331, 64)),
        ("PETS", "500", (206, 30, 402, 457, 105)),
        ("", "2,000", (279, 28, 349, 419, 125)),
        None,
        ("CrazyMARL", "", (487, 1, 712, 0, 0)),
        ("TD-MPC2", "", (0, 0, 1026, 159, 15)),
        ("DreamerV3", "", (0, 0, 1013, 112, 75))]


# Same 17 rows, five categories, source counts and original colors.
from matplotlib.transforms import blended_transform_factory
fig, ax = plt.subplots(figsize=(3.5,4.5))
fig.subplots_adjust(left=0.455,right=0.955,top=0.825,bottom=0.16)
y=1.3
groups=[(0,"Reference rows")]
plot_records=[]
method_labels=[]
budget_labels=[]
group_index=0
current_name=""
for row in rows:
    if row is None:
        group_index+=1
        heading_y=y+0.4
        groups.append((heading_y,"Learned world models" if group_index==1 else "Independent policies"))
        y=heading_y+1.3
        current_name=""
        continue
    name,episodes,counts=row
    left=0
    widths=[]
    for (_,colour),count in zip(outcomes,counts):
        width=100*count/sum(counts)
        ax.barh(y,width,left=left,height=0.72,color=colour,edgecolor="white",linewidth=0.5)
        widths.append(width)
        left+=width
    # Every bar gets a method label at its OWN centre, including both budgets.
    if name:
        current_name=name
    method_labels.append(ax.annotate(current_name,(0.045,y),xycoords=("figure fraction","data"),
                    ha="left",va="center",color=INK,fontsize=7.0))
    if episodes:
        budget_labels.append(ax.annotate(episodes,(0.420,y),xycoords=("figure fraction","data"),
                    ha="right",va="center",color=INK_2,fontsize=7.0))
    plot_records.append(dict(name=name,episodes=episodes,counts=list(counts),total=sum(counts),
                             percentages=widths,y=y))
    y+=1
for position,label in groups:
    # Group title is a separate header row, not a label for the first bar.
    ax.plot([0.015,0.955],[position-0.60,position-0.60],
            transform=blended_transform_factory(fig.transFigure,ax.transData),color=REF,linewidth=0.55,clip_on=False)
    ax.annotate(label,(0.015,position),xycoords=("figure fraction","data"),ha="left",va="center",
                color=INK,fontsize=7.6,fontweight="bold")
fig.text(0.045,0.855,"Method",va="center",color=INK_2,fontsize=7.2)
fig.text(0.382,0.855,"Training\nepisodes",ha="center",va="center",color=INK_2,fontsize=7.0,linespacing=1.1)
ax.set_yticks([])
ax.set_ylim(y-0.4,-1.0)
ax.set_xlim(0,100)
ax.set_xticks(range(0,101,25))
ax.set_xlabel("All test episodes (%)",color=INK_2,labelpad=5)
for side in ("top","right","left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(REF)
ax.spines["bottom"].set_linewidth(0.6)
ax.tick_params(colors=INK_2,width=0.6,length=2.5)
# Arrange the legend row-wise in the same order as the stacked categories.
# Matplotlib fills a two-column legend down columns, hence the permutation.
handles=[plt.Rectangle((0,0),1,1,color=colour) for _,colour in outcomes]
order=[0,2,4,1,3]
fig.legend([handles[i] for i in order],[outcomes[i][0] for i in order],loc="upper center",ncol=2,frameon=False,
           bbox_to_anchor=(0.5,1.0),handlelength=1.0,handletextpad=0.4,columnspacing=1.0,labelcolor=INK_2,
           fontsize=7.2)
fig.text(0.015,0.025,"400 flights per model; 1,200 for three pooled seeds.",color=INK_2,fontsize=6.4)
out=sys.argv[1]
fig.savefig(out+".pdf")
fig.savefig(out+".png",dpi=300)
print("wrote",out+".pdf",out+".png")

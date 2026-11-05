"""Fig. 7: the primary statistic rho(L2, J | L1) with its 95% interval, across all 30 learned models, in each test
condition, and with one row's six models left out.

The first two are "primary_set" and "per_condition" in data/uav_arena/results/primary_analysis.json, printed in
uav_arena/README.md, "Pre-registered analysis" (uav-arena 0fdb022); the third, outside ANALYSIS.md, is
figures/src/rows_check.py on the same 10,000 layout resamples. A filled marker means the interval lies below zero.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7.5,
                     "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7})

INK, INK_2, GRID, REF, BAND = "#0b0b0b", "#52514e", "#e6e5e1", "#8a8984", "#ecebe7"
# (estimate, low, high)
pooled = (-0.3904, -0.6023, -0.2739)
groups = [("Pillars alone", [("0 ms", (-0.5188, -0.6234, -0.0623)), ("50 ms", (-0.5663, -0.6569, -0.2509)),
                             ("100 ms", (-0.5014, -0.6176, -0.1860)), ("200 ms", (-0.0335, -0.2578, 0.3045))]),
          ("With moving spheres", [("0 ms", (-0.2847, -0.4696, 0.1282)), ("50 ms", (-0.4311, -0.5723, 0.0608)),
                                   ("100 ms", (-0.2321, -0.3991, 0.0159)), ("200 ms", (-0.1620, -0.3648, 0.0728))]),
          ("One row left out, outside the plan",
           [("without JEPA full", (-0.6159, -0.7862, -0.4768)), ("without JEPA no_cross", (-0.5349, -0.7229, -0.4291)),
            ("without JEPA flat", (0.1947, -0.1639, 0.2877)), ("without W-ar", (-0.5948, -0.7310, -0.4689)),
            ("without PETS", (0.0636, -0.4172, 0.2329))])]


from matplotlib.transforms import blended_transform_factory
fig, ax = plt.subplots(figsize=(3.5, 4.4))
fig.subplots_adjust(left=.46, right=.97, top=.975, bottom=.20)
ax.axvline(0, color=INK_2, lw=.7, zorder=1)
records = []
label_artists = []

def label(y, text, heading=False, subheading=False):
    artist = ax.annotate(text, (.015 if heading else .04, y),
                        xycoords=('figure fraction', 'data'), va='center', ha='left',
                        color=INK if heading else INK_2,
                        fontsize=7.4 if heading else 7.0,
                        fontweight='bold' if heading else 'normal')
    label_artists.append(artist)

def rule(y):
    ax.plot([.015,.97], [y,y], color=REF, lw=.5, clip_on=False,
            transform=blended_transform_factory(fig.transFigure, ax.transData))

def interval(y, text, values, primary=False):
    estimate, low, high = values
    line, = ax.plot([low,high], [y,y], color=INK_2,
                    lw=1.4 if primary else 1.0, solid_capstyle='butt', zorder=2)
    point = ax.scatter(estimate,y,s=25 if primary else 19,
                       color=INK if high < 0 else 'white',edgecolor=INK,linewidth=.9,zorder=3)
    label(y,text)
    records.append((text,values,y,line,point,label_artists[-1]))

label(0,'Primary analysis (pre-registered)',heading=True)
interval(1.2,'All 30 models',pooled,primary=True)
rule(2.2)
label(3.0,'By test condition',heading=True)
label(4.15,'Pillars alone')
for y,(text,values) in zip([5.3,6.3,7.3,8.3],groups[0][1]):
    interval(y,'    '+text,values)
label(9.6,'With moving spheres')
for y,(text,values) in zip([10.75,11.75,12.75,13.75],groups[1][1]):
    interval(y,'    '+text,values)
rule(14.75)
label(15.55,'Exploratory: one model row left out',heading=True)
for y,(text,values) in zip([16.85,17.85,18.85,19.85,20.85],groups[2][1]):
    interval(y,text,values)

ax.set_yticks([])
ax.set_ylim(21.7,-.7)
ax.set_xlim(-1,.5)
ax.set_xticks([-1,-.5,0,.5],['−1','−0.5','0','0.5'])
ax.set_xlabel(r'Partial Spearman $\rho(L_2,J\mid L_1)$',color=INK_2,labelpad=5)
ax.grid(True,axis='x',color=GRID,lw=.5)
ax.set_axisbelow(True)
for side in ('top','right','left'):
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color(REF)
ax.spines['bottom'].set_linewidth(.6)
ax.tick_params(colors=INK_2,width=.6,length=2.5)
fig.text(.015,.083,'Points: estimates; lines: 95% bootstrap intervals.',fontsize=6.8,color=INK_2)
fig.text(.015,.047,'Filled: interval below 0; open: interval includes 0.',fontsize=6.8,color=INK_2)
out = sys.argv[1]
fig.savefig(out+'.pdf')
fig.savefig(out+'.png',dpi=300)

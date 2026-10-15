##MAT2007

##-----------Study Techniques Project Data Analysis------------##

## Project RQ: Do active or passive study techniques yield the highest grades?

# DATA INFO:
#  pre_ss_"variables" : study techniques students said they used (0 = no, 1 = yes)
#  grade_rec : recoded course grade 1-13 (1 = F, 13 = A+, A - being the highest grade)

# rm(list = ls()) #clear environemnt

from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

DATA_DIR = Path(__file__).resolve().parent

##--------------1. Importing data----------------##

#Reading the csv files
year1 = pd.read_csv(DATA_DIR / 'Year1studystrat.csv', encoding='latin1')
year2 = pd.read_csv(DATA_DIR / 'Year2studystrat.csv', encoding='latin1')

# Adding a column that tells us which cohort a student belongs to.
year1['year'] = 1
year2['year'] = 2

# Converting both column to text "as.character", just so we can combine the 2 data sets, we wont use these variable for the analysis itself
# Keep missing values missing.
year1['post_ss_selftest'] = year1['post_ss_selftest'].astype('string')
year2['post_ss_selftest'] = year2['post_ss_selftest'].astype('string')

# Combine the 2 data sets
dat = pd.concat([year1, year2], ignore_index=True)

##--------------2. Check how often each study techniques is used---------------##
techniques = [
    'pre_ss_cram',       # cramming
    'pre_ss_selftest',   # self-testing
    'pre_ss_reread',     # rereading
    'pre_ss_videos',     # watching videos
    'pre_ss_condense',   # condensing notes
    'pre_ss_diag',       # drawing diagrams
    'pre_ss_high',       # highlighting
    'pre_ss_flashc',     # flashcards
    'pre_ss_friends',    # studying with friends
    'pre_ss_recopy',     # recopying notes
    'pre_ss_recopymem',  # recopying notes from memory
    'pre_ss_other'       # other technique
]

# Because the variables are coded 0/1, the mean is the proportion
# of students who selected each technique.
usage = dat[techniques].mean(skipna=True)  # colMeans() calculates the mean of every column

# Convert to percentages and sort from most to least used
usage_percent = (usage * 100).sort_values(ascending=False)

# Print results
print('\nTechnique usage (%):\n', usage_percent)

##-----------------3. Check how many techniques each student selected-----------##

# The students were supposed to pick 3 most used techniques
# Variables are 0/1, the sum = number of techniques selected
dat['n_techniques'] = dat[techniques].sum(axis=1, skipna=True) ## rowSums - adding up rows

# Number of students selecting 0, 1, 2, 3, etc. techniques
print('\nNumber selected:\n', dat['n_techniques'].value_counts().sort_index())

# Percentages
print('\nPercentages:\n', dat['n_techniques'].value_counts(normalize=True).sort_index() * 100)

# Keep only students who selected exactly 3 techniques.
# nrow() before and after shows how many students are removed.
n_before = len(dat)
dat = dat.loc[dat['n_techniques'] == 3].copy()
n_after = len(dat)
print('\nNumber removed:', n_before - n_after)    # number of students removed
print('\nStudents per cohort:\n', dat['year'].value_counts().sort_index())       # students per cohort after filtering

## -----------------4. Define Active and Passive techniques-----------##

# Active techniques:
# self-testing, flashcards, recopying from memory, condensing, diagrams, studying with friends
active_vars = [
    'pre_ss_selftest', 'pre_ss_flashc', 'pre_ss_recopymem',
    'pre_ss_condense', 'pre_ss_diag', 'pre_ss_friends'
]

# Passive techniques:
# rereading, highlighting, videos, recopying, cramming
passive_vars = [
    'pre_ss_reread', 'pre_ss_high', 'pre_ss_videos',
    'pre_ss_recopy', 'pre_ss_cram'
]

# Count active and passive techniques per student in new columns
dat['active_count'] = dat[active_vars].sum(axis=1, skipna=True)
dat['passive_count'] = dat[passive_vars].sum(axis=1, skipna=True)

# Active-passive orientation score = active count - passive count.
# higher score = more active
# lower score = more passive
#  0 = balanced (or none of the techniques)
dat['active_passive_score'] = dat['active_count'] - dat['passive_count']

# active techniques counts as +1
# passive techniques counts as -1
# Example: active , active , passive = 1 (score)
# Example: passive , passive , passive = -3

## ------------5. Inspect active/passive scores--------##

# summary() gives min, quartiles, median, mean and max of a variable.
for v in ['active_count', 'passive_count', 'active_passive_score']:
    print(f'\n{v}:\n', dat[v].describe())

# How many students have each score value?
print('\nScore frequencies:\n', dat['active_passive_score'].value_counts().sort_index())

## -----------------6. Main Analysis - Linear Regression------------------###
# Is a stronger active rather than passive study profile associated with higher grades?

##  lm() - linear regression model
##  active_passive_score -  the average change in grade (on the 1-13 scale) for a
##  1-point higher active-passive score, with cohort (year) as a constant (nr. answers the RQ)
##  year (Year2) - average grade difference between Year 2 and Year 1
##  at the same score. It controls for cohort differences.
model_primary = smf.ols('grade_rec ~ active_passive_score + year', data=dat).fit()

# Regression results
# summary() results: the estimate, its standard error, a t-test of H0: "coefficient = 0" (with p-value),
# plus R-squared and F-test of the whole model at the bottom (H0: all coefficients are 0).
# A p-value below 0.05 means the association is statistically significant.
print('\nPRIMARY MODEL\n', model_primary.summary())

# 95% confidence intervals
# confint() gives the 95% confidence interval of each coefficient.
# If the interval does not contain 0, the effect is significant at 5%.
print(model_primary.conf_int())

# R-squared = proportion of the variation in grades explained by the model.
# Adjusted R-squared corrects for the number of predictors.
print('R-squared:', model_primary.rsquared)
print('Adjusted R-squared:', model_primary.rsquared_adj)

# nobs() = number of students actually used in the model
print('N:', int(model_primary.nobs))

# Overall model test
# anova() on a model gives a SEQUENTIAL analysis of variance table: an F-test
# for each term, added one after the other in the order of the formula
# ( score first, then year).
print(sm.stats.anova_lm(model_primary, typ=1))

##----------------7. SECONDARY ANALYSIS---------------------##
# Which out of the 6 most used techniques are associated with higher grades?
# Included top 6 most used techniques:
# - self-testing
# - rereading
# - condensing
# - videos
# - diagrams
# - studying with friends

# Multiple linear regression: all six techniques are entered TOGETHER, so
# each coefficient is the difference in mean grade between students who used
# that technique (1) and those who did not (0). Adjusted for other techniques and year.
model_secondary = smf.ols(
    'grade_rec ~ pre_ss_selftest + pre_ss_reread + pre_ss_condense + '
    'pre_ss_videos + pre_ss_diag + pre_ss_friends + year', data=dat
).fit()

# Regression results
print('\nSECONDARY MODEL\n', model_secondary.summary())

# 95% confidence intervals
print(model_secondary.conf_int())

# R-squared and adjusted R-squared
print('R-squared:', model_secondary.rsquared)
print('Adjusted R-squared:', model_secondary.rsquared_adj)

# Overall model test
print(sm.stats.anova_lm(model_secondary, typ=1))


#----------------------------------------------------------------------------------------
# PLOTS

#   Plot 1: grade by active/passive score (answers the RQ)
#   Plot 2: mean grade of students who chose each technique

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t as student_t

OUT = Path(__file__).resolve().parent

# Readable names for the technique variables.
tech_labels = {
    'pre_ss_cram': 'Cramming',
    'pre_ss_selftest': 'Self-testing',
    'pre_ss_reread': 'Rereading',
    'pre_ss_videos': 'Videos',
    'pre_ss_condense': 'Condensing notes',
    'pre_ss_diag': 'Diagrams',
    'pre_ss_high': 'Highlighting',
    'pre_ss_flashc': 'Flashcards',
    'pre_ss_friends': 'Studying with friends',
    'pre_ss_recopy': 'Recopying notes',
    'pre_ss_recopymem': 'Recopying from memory',
    'pre_ss_other': 'Other'
}

# Colours for the three groups
col_active = '#FF3E96'   # pink   = active techniques 
col_passive = '#1E90FF'   # blue = passive techniques
col_other = '#7F7F7F'    # grey   = not classified (only "other")

# Function: gives each technique the colour of its group.
#   v = vector of technique variable names, e.g. "pre_ss_cram"
#   %in% checks whether each name is in active_vars / passive_vars
#   ifelse() picks a colour per technique: active -> pink, passive -> blue,
#   anything else -> grey
def group_colour(v):
    return [col_active if x in active_vars else col_passive if x in passive_vars else col_other for x in v]


plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.titlesize': 12})

# ----------------------------------------------------------------------
# PLOT 1: GRADE BY ACTIVE/PASSIVE SCORE
# Boxplot
# "Do students with a more active profile get higher grades?"

# Only the four common scores are plotted.
# The scores 0 and +2 are omitted - have only 12 and 6 students, too few for a meaningful box.
scores_shown = [-3, -1, 1, 3]

# Keep only the students with one of these scores.
d1 = dat.loc[dat['active_passive_score'].isin(scores_shown)].copy()

# factor() turns the score into a categorical variable with a fixed order,
# so the boxes appear from -3 (passive) to +3 (active).
d1['score_f'] = pd.Categorical(d1['active_passive_score'], categories=scores_shown, ordered=True)

# tapply(values, groups, function) applies a function to the grades of each
# group. as.vector() removes the extra attributes tapply adds.
groups1 = [d1.loc[d1['score_f'] == s, 'grade_rec'].dropna().to_numpy() for s in scores_shown]
n_per = np.array([len(g) for g in groups1])  # students per score
mean_per = np.array([np.mean(g) for g in groups1])  # mean grade per score
sd_per = np.array([np.std(g, ddof=1) for g in groups1])  # standard deviation

# 95% confidence interval of each mean:
#   mean +/- t * standard error, with standard error = sd / sqrt(n)
#   qt(0.975, df) gives the t-value for a 95% interval (df = n - 1)
se_per = sd_per / np.sqrt(n_per)
ci_low = mean_per - student_t.ppf(.975, n_per - 1) * se_per
ci_high = mean_per + student_t.ppf(.975, n_per - 1) * se_per

# boxplot(formula, data) draws one box per group:
#   the thick line = median, the box = middle 50% of students,
#   the whiskers = range of typical values, circles = outliers.
# names = labels under the boxes (score and number of students).
fig, ax = plt.subplots(figsize=(2000 / 300, 1500 / 300), dpi=300)
ax.boxplot(groups1, patch_artist=True, widths=.5,
           boxprops=dict(facecolor='#E5E5E5', edgecolor='#666666'),
           medianprops=dict(color='#333333', linewidth=1.5),
           whiskerprops=dict(color='#666666'), capprops=dict(color='#666666'),
           flierprops=dict(marker='o', markersize=3, markerfacecolor='none', markeredgecolor='#666666'))
ax.set_xticks(range(1, 5), [f'{s}\n(n = {n})' for s, n in zip(scores_shown, n_per)])
ax.set_ylim(1, 13)
ax.set_xlabel('Active-passive score  (-3 = fully passive, +3 = fully active)')
ax.set_ylabel('Course grade (1-13)')
ax.set_title('Grade by active/passive study profile')

# arrows() draws the error bars. code = 3 puts a cap on both ends and
# angle = 90 makes the caps flat (so they look like error-bar ends).
ax.errorbar(range(1, 5), mean_per, yerr=[mean_per - ci_low, ci_high - mean_per],
            fmt='none', ecolor='#912CEE', capsize=4, elinewidth=2)

# points() adds the mean as a purple diamond (pch = 18) on top of each box.
ax.scatter(range(1, 5), mean_per, marker='D', s=55, color='#912CEE', zorder=4)

# legend() explains symbol.
ax.legend([Line2D([0], [0], marker='D', linestyle='none', color='#912CEE')],
          ['Mean with 95% CI'], loc='lower right', frameon=False)

# Add extra space at the bottom for the x-axis labels
fig.subplots_adjust(bottom=0.25)

# Automatically adjust spacing
fig.tight_layout()

# Save the complete figure without cutting off labels
fig.savefig(OUT / 'plot1.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.2)

# Add extra space at the bottom for the x-axis labels
fig.subplots_adjust(bottom=0.25)

# Automatically adjust spacing
fig.tight_layout()

# Save the complete figure without cutting off labels
fig.savefig(OUT / 'plot1.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.2)

# Add extra space at the bottom for the x-axis labels
fig.subplots_adjust(bottom=0.25)

# Automatically adjust spacing
fig.tight_layout()

# Save the complete figure without cutting off labels
fig.savefig(OUT / 'plot1.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.2)


# Add extra space at the bottom for the x-axis labels
fig.subplots_adjust(bottom=0.25)

# Automatically adjust spacing
fig.tight_layout()

# Save the  figure without cutting off plot
fig.savefig(OUT / 'plot1_grade_by_score.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.2)

plt.close(fig)


# ----------------------------------------------------------------------
# PLOT 2: MEAN GRADE OF STUDENTS WHO CHOSE EACH TECHNIQUE
# Descriptive plot:
# Average grade for each technique, plus how many students that is.

# For every technique: take the grades of the students who selected it
# (value 1), then calculate n, % of all students, mean and 95% CI.
# lapply() repeats the calculation for each technique and do.call(rbind, ...)
# stacks the results into one table.
rows = []
for v in techniques:
    g = dat.loc[dat[v] == 1, 'grade_rec'].to_numpy()  # grades of students who chose technique v
    m = np.mean(g)
    se = np.std(g, ddof=1) / np.sqrt(len(g))  # standard error of the mean
    rows.append({'var': v, 'n': len(g), 'pct': 100 * len(g) / len(dat),  # % of all students who chose it
                 'mean': m, 'low': m - student_t.ppf(.975, len(g) - 1) * se,
                 'high': m + student_t.ppf(.975, len(g) - 1) * se})
res3 = pd.DataFrame(rows)

# Sort from highest to lowest mean grade.
res3 = res3.sort_values('mean', ascending=False, kind='stable').reset_index(drop=True)
res3['y'] = np.arange(len(res3), 0, -1)
res3['col'] = group_colour(res3['var'])

# y-axis labels: name, number of students and percentage.
labels3 = [f'{tech_labels[r.var]}  (n = {r.n}, {round(r.pct)}%)' for r in res3.itertuples()]

# Overall mean grade of all students, used as a reference line.
overall_mean = dat['grade_rec'].mean()

fig, ax = plt.subplots(figsize=(2200 / 300, 1600 / 300), dpi=300)
fig.subplots_adjust(left=.43, right=.96, bottom=.14, top=.88)
ax.set_xlim(res3['low'].min(), res3['high'].max())
ax.set_ylim(.5, len(res3) + .5)
ax.set_xlabel('Mean course grade (95% CI)')
ax.set_title('Mean grade of students who chose each technique')

# Dashed line = mean grade of all students.
ax.axvline(overall_mean, linestyle='--', color='#666666')

for r in res3.itertuples():
    ax.hlines(r.y, r.low, r.high, color=r.col, linewidth=2)
    ax.plot(r.mean, r.y, 'o', color=r.col, markersize=6)
ax.set_yticks(res3['y'], labels3)
ax.tick_params(axis='y', length=0)

ax.legend([Line2D([0], [0], marker='o', linestyle='none', color=c) for c in [col_active, col_passive, col_other]] +
          [Line2D([0], [0], linestyle='--', color='#666666')],
          ['Active', 'Passive', 'Not classified', 'Mean of all students'], loc='lower right', frameon=False)
fig.savefig(OUT / 'plot2_mean_grade_per_technique.png', dpi=300)
plt.close(fig)

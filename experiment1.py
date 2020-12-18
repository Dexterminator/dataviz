import seaborn as sb
import matplotlib.pyplot as plt

from utils import show, close_pdf

sb.set_style('whitegrid')

# %%
crash_df = sb.load_dataset('car_crashes')

# %%
show(sb.displot(crash_df['not_distracted'], bins=25), 'Not distracted')

# %%
show(sb.jointplot(x='speeding', y='alcohol', data=crash_df), 'Speeding vs. alcohol')
show(sb.jointplot(x='speeding', y='alcohol', data=crash_df, kind='hex'), 'Speeding vs. alcohol hotspots')

# %%
show(sb.pairplot(crash_df), 'Crashes pairplot')

# %%
tips_df = sb.load_dataset('tips')
tips_df

# %%
plt.figure()
show(sb.pairplot(tips_df, hue='sex'))

# %%
show(sb.rugplot(tips_df['tip']))

# %%
show(sb.jointplot(x='speeding', y='alcohol', data=crash_df, kind='reg'), 'Speeding vs. alcohol reg')

# %%
show(sb.barplot(x='sex', y='total_bill', data=tips_df))

# %%
show(sb.countplot(x='sex', data=tips_df))

# %%
boxplot = sb.boxplot(x='day', y='total_bill', data=tips_df, hue='sex')
plt.legend(loc=3)
show(boxplot)

# %%
show(sb.violinplot(x='day', y='total_bill', data=tips_df, hue='sex', split=True))

# %%
show(sb.stripplot(x='day', y='total_bill', data=tips_df, jitter=True, hue='sex', dodge=True))

# %%
sb.violinplot(x='day', y='total_bill', data=tips_df)
show(sb.swarmplot(x='day', y='total_bill', data=tips_df, color='white'))

# %%
crash_mx = crash_df.corr()
show(sb.heatmap(crash_mx, annot=True, cmap='Blues'))

# %%
flights_df = sb.load_dataset('flights')
flights_pivot = flights_df.pivot_table(index='month', columns='year', values='passengers')
show(sb.heatmap(flights_pivot, cmap='Blues'))

# %%
iris_df = sb.load_dataset('iris')
iris_df.pop('species')
show(sb.clustermap(iris_df))

# %%
iris_df = sb.load_dataset('iris')
iris_g = sb.PairGrid(iris_df, hue='species')
iris_g.map_diag(plt.hist)
# iris_g.map_offdiag(plt.scatter)
iris_g.map_upper(plt.scatter)
iris_g.map_lower(sb.kdeplot)
show(iris_g, 'Iris pairgrid')

# %%
iris_df = sb.load_dataset('iris')
show(sb.pairplot(iris_df, hue='species'), 'Iris scatterplot')

# %%
tips_fg = sb.FacetGrid(tips_df, col='time', hue='smoker', height=4,
                       aspect=1.3)
tips_fg.map(plt.scatter, 'total_bill', 'tip')
show(tips_fg)

# %%
plt.figure(figsize=(8, 6))
show(sb.lmplot(x='total_bill', y='tip', hue='sex', data=tips_df,
          markers=['o', '^'], scatter_kws={'s': 100, 'linewidth': 0.5,
                                           'edgecolor': 'white'}))

# %%
show(sb.lmplot(x='total_bill', y='tip', col='sex', row='time', data=tips_df))

# %%
close_pdf()

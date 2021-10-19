import pandas as pd
df = pd.DataFrame({'A':[5,91,3],'B':[90,15,66],'C':[93,27,3]})
print(df.corr())
print(df.corr('spearman'))
print(df.corr('kendall'))
df2 = pd.DataFrame({'A':[7,93,5],'B':[88,13,64],'C':[93,27,3]})
print(df2.corr())
print(df2.corr('spearman'))
print(df2.corr('kendall'))
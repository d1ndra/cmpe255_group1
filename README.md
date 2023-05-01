# Data Mining code module

Install using `pip install git+https://github.com/d1ndra/cmpe255_group1.git`

Usage:

```python
from cmpe255gp1.kmeans import KMeans
X = df #some dataframe
model = KMeans(num_cluster=3, max_iter=30)
model.fit(X)
```
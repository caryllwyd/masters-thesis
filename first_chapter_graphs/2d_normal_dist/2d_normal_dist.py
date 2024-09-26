from sklearn.datasets import load_wine
import pandas as pd

load_data = load_wine(as_frame=True)
wine_df = pd.DataFrame(load_data.data, columns=load_data.feature_names)
alcohol_column = wine_df['alcohol']
print(alcohol_column)
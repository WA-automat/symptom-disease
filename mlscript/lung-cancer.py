import pandas as pd

df = pd.read_csv('../data/lung-cancer.csv')

if __name__ == '__main__':
    print(df.info())

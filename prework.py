import json
import jieba
import pandas as pd

train_zh_df = pd.read_csv('./data/symptom-disease-train-dataset-zh.csv', index_col=0)
test_zh_df = pd.read_csv('./data/symptom-disease-test-dataset-zh.csv', index_col=0)

if __name__ == '__main__':
    # 转换数据类型并排序
    train_zh_df['label'] = train_zh_df['label'].astype(int)
    test_zh_df['label'] = test_zh_df['label'].astype(int)
    train_zh_df_sorted = train_zh_df.sort_values(by='label')
    test_zh_df_sorted = test_zh_df.sort_values(by='label')

    # 去除仅有一条训练数据的label
    counts = train_zh_df_sorted['label'].value_counts()
    cnt = 0
    for label, count in counts.items():
        if count > 1:
            cnt += 1
        else:
            train_zh_df_sorted = train_zh_df_sorted[train_zh_df_sorted['label'] != label]
            test_zh_df_sorted = test_zh_df_sorted[test_zh_df_sorted['label'] != label]

    # 转换为对应病症
    with open("./data/mapping-zh.json", 'r', encoding='utf-8') as f:
        mp = json.load(f)
    rev_mp = {value: key for key, value in mp.items()}
    train_zh_df_sorted['label'] = train_zh_df_sorted['label'].apply(lambda x: rev_mp[x])
    test_zh_df_sorted = test_zh_df_sorted[test_zh_df_sorted['label'].isin(counts.keys())]
    test_zh_df_sorted['label'] = test_zh_df_sorted['label'].apply(lambda x: rev_mp[x])

    # 生成新的mapping
    labels = train_zh_df_sorted['label'].unique()
    labels = [s.strip() for s in labels]
    labels = list(set(labels))
    new_mapping = {label: i for i, label in enumerate(labels)}
    # with open("./data/mapping-zh-new.json", 'w', encoding='utf-8') as f:
    #     json.dump(new_mapping, f, ensure_ascii=False, indent=4)
    train_zh_df_sorted['label'] = train_zh_df_sorted['label'].str.strip()
    test_zh_df_sorted['label'] = test_zh_df_sorted['label'].str.strip()
    df = pd.DataFrame({'索引': list(new_mapping.values()), '病症': list(new_mapping.keys())})
    df.sort_values(by='索引')
    # df.to_excel('./data/病症及其对应的建议.xlsx', index=False)

    # 使用新的mapping并排序
    train_zh_df_sorted['label'] = train_zh_df_sorted['label'].apply(lambda x: new_mapping[x])
    test_zh_df_sorted['label'] = test_zh_df_sorted['label'].apply(lambda x: new_mapping[x])
    train_zh_df_sorted = train_zh_df_sorted.sort_values(by='label')
    test_zh_df_sorted = test_zh_df_sorted.sort_values(by='label')

    train_zh_df_sorted.drop_duplicates(subset='text', inplace=True)
    test_zh_df_sorted.drop_duplicates(subset='text', inplace=True)

    # 保存结果
    # train_zh_df_sorted.to_csv('./data/symptom-disease-train-dataset-zh-new.csv', index=False)
    # test_zh_df_sorted.to_csv('./data/symptom-disease-test-dataset-zh-new.csv', index=False)

    # 构建词表并保存
    with open('./data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
    stopwords = [line.rstrip('\n') for line in stopwords]
    sentences = train_zh_df_sorted['text'].values
    words = [word for sentence in sentences for word in jieba.cut(sentence) if word not in stopwords]
    words = list(set(words))
    words.insert(0, '<bos>')
    words.append('<pad>')
    words.append('<unk>')
    words.append('<eos>')
    vocab = {value: index for index, value in enumerate(words)}
    # with open('./data/vocab.json', 'w', encoding='utf-8') as f:
    #     json.dump(vocab, f, ensure_ascii=False, indent=4)

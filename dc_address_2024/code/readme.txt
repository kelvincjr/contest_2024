1. 预测：run.py 主函数，主要调用final_test.py里面的功能，参数：所需预测的文件夹路径，预测结果保存文件路径
2. 训练：ner_arknlp.py，参数：model_name_or_path存在bert预训练模型的路径，num_train_epochs epoch数
训练结果保存在model_save目录下

nohup python ner_arknlp.py --model_name_or_path ./bert_model/ --num_train_epochs 10 > nohup_train_bert_epoch10.out 2>&1 &

3. 工具：tool.py，功能包括训练数据自动生成，补充数据自动搜索和下载
但因为工具程序还没有时间很好地整理代码，结构有点乱，请见谅。

4. 其它文件：langconv.py和zh_wiki.py主要提供繁体简体转换功能，trie_lib.py和trie_lib目录主要提供trie树查找功能。

5. 目录bert_model存放bert预训练模型，目录model_save保存训练出来最优的地址元素实体识别模型。

6. 目录data存放使用到的数据：

train_data.csv：赛题提供的原始训练数据
test_data.csv：赛题提供的初赛测试数据
train_data_with_label_new.json：程序自动生成的训练实体标注数据，用于模型训练
siming_district.txt：网上公开的厦门思明区街道和道路数据，作为补充数据
keyword_amap.json：高德API自动搜集的厦门POI数据（主要是思明区）

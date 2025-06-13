import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import pandas as pd 
from final_test import ner_predict, gen_submission_final, get_addr_pool
'''
def main(to_pred_dir, result_save_path):
    df_dict = pd.read_csv(os.path.join(to_pred_dir, 'final_test/决赛标准地址池.csv'))
    df_test = pd.read_csv(os.path.join(to_pred_dir, 'final_test/决赛测试集.csv'))
    # 注意这里只是随机的样例，选手可以替换成自己预测的结果
    df_sub = pd.DataFrame()
    df_sub['id'] = df_test['id']
    df_sub['address'] = df_dict['标准地址池'][0]
    df_sub.iloc[0,1] = '福建省厦门市思明区莲前西路733号602室'
    # 将预测结果保存到指定位置
    df_sub.to_csv(result_save_path, index=None)
'''
def main(to_pred_dir, result_save_path):
    df_dict = pd.read_csv(os.path.join(to_pred_dir, 'final_test/决赛标准地址池.csv'))
    df_test = pd.read_csv(os.path.join(to_pred_dir, 'final_test/决赛测试集.csv'))

    addr_pool = get_addr_pool(os.path.join(to_pred_dir, 'final_test/决赛标准地址池.csv'))
    ner_predict(df_test)
    results = gen_submission_final(addr_pool)
    
    # 注意这里只是随机的样例，选手可以替换成自己预测的结果
    df_sub = pd.DataFrame()
    df_sub['id'] = df_test['id']
    df_sub['address'] = df_dict['标准地址池'][0]
    #df_sub.iloc[0,1] = '福建省厦门市思明区莲前西路733号602室'
    count = 0
    for item in results:
        id_ = item[0]
        address = item[2]
        df_sub.iloc[count,1] = address
        count += 1
    # 将预测结果保存到指定位置
    df_sub.to_csv(result_save_path, index=None)
    '''
    with open(result_save_path, "a+", encoding="utf-8") as f:
        f.write("{},{}\n".format("id","address"))
        for item in results:
            id_ = item[0]
            address = item[2]
            f.write("{},{}\n".format(id_,address))
    '''
if __name__ == "__main__":
    # ！！！以下内容不允许修改，修改会导致评分出错
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径，已指定格式为csv
    main(to_pred_dir, result_save_path)

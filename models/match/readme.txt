## Movielens

使用ml-1m数据集，使用其中原始特征7个user特征`'user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip',"cate_id"`，2个item特征`"movie_id", "cate_id"`，一共9个sparse特征。

- 构造用户观看历史特征``hist_movie_id``，使用`mean`池化该序列embedding
- 使用随机负采样构造负样本 (sample_method=0)，内含随机负采样、word2vec负采样、流行度负采样、Tencent负采样等多种方法
- 将每个用户最后一条观看记录设置为测试集
- 原始数据下载地址：https://grouplens.org/datasets/movielens/1m/
- 处理完整数据csv下载地址：https://cowtransfer.com/s/5a3ab69ebd314e

以下指标使用example中的相同参数在ml-1m上测试得到
| Model\Metrics | Hit@100 | Recall@100 | Precision@100 |
|---------------|---------|------------|---------------|
| DSSM          | 14.74%  | 14.74%     | 0.15%         |
| YoutubeDNN    | 6.27%   | 6.27%      | 0.06%         |
| FacebookDSSM  | 2.85%   | 2.85%      | 0.03%         |
| YoutubeSBC    | 17.83%  | 17.83%     | 0.18%         |

> 解释：FacebookDSSM 最差原因是pairwise 分布很难拟合真实分布，YoutubeDNN 较差原因是item 缺少特征，DSSM 次之时因为负样本只选择一个，拟合真实分布有损失

## 双塔模型对比

| 模型         | 学习模式   | 损失函数  | 样本构造                                                     | label                              |
| ------------ | ---------- | --------- | ------------------------------------------------------------ | ---------------------------------- |
| DSSM         | point-wise | BCE       | 全局负采样，一条负样本对应label 0                            | 1或0                               |
| YoutubeDNN   | list-wise  | CE        | 全局负采样，每条正样本对应k条负样本                          | 0（item_list中第一个位置为正样本） |
| YoutubeSBC   | list-wise  | CE        | Batch内随机负采样，每条正样本对应k条负样本，加入采样权重做纠偏处理 | 0（item_list中第一个位置为正样本） |
| FacebookDSSM | pair-wise  | BPR/Hinge | 全局负采样，每条正样本对应1个负样本，需扩充负样本item其他属性特征 | 无label                            |

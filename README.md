
# Introduction 导言
score transform model
分数转换模型
models include: zhejiang/shanghai/shandong/beijing/tianjian level score model
模型中包括： 浙江 上海 山东 北京 天津 的等级分数转换模型
also include: Zscore, Tscore
也包括： Z分数 T分数 转换
even include: level score model that Tao Baiqiang designed in his article.
也包括： 陶百强提出的等级分数转换模型

designer, Wang Xichang, wxc1964@126.com
设计者，姓名，信箱 

# Level Score Model 等级分数模型

(1) 用于浙江、上海、北京、天津等省新高考中的等级赋分模型。

the level score model, that is used to dispatch score value in Zhejiang, Shanghai, Beijing, Tianjin New High Test。

等级分数模型通过以下步骤完成：
1. 指定比例，划分原始分数为各个等级区间
2. 每个区间指定等级
3. 每个等级指定分数
4. 从各个考生的原始分数，通过其所在区间得到等级，通过等级得到等级分数。

stm的计算过程：
1. 计算分数字段的分段表，每个字段包括四个值：分段值（某一分值），对应该分段值的的人数，百分比，累积百分比，分段值按照顺序（默认从高到低）
2. 在分段表中计算，每个分段值对应的等级，等级分数
3. 使用分段表计算每个考生该分数字段的等级分数

(2) 用于山东省等级分数转换的模型
the model used in Shandong New High-Test Project


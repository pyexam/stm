
# Introduction 导言

score transform model

分数转换模型

models include: zhejiang/shanghai/shandong/beijing/tianjian level score model

模型中包括： 浙江 上海 山东 北京 天津 的等级分数转换模型

also include: Zscore, Tscore，level score model that Tao Baiqiang designed in his article.

也包括： Z分数、T分数转换，陶百强提出的等级分数转换模型

designer, Wang Xichang, wxc1964@126.com

设计者，姓名，信箱 

# Level Score Model 等级分数模型

用于浙江、上海、北京、天津、山东新高考中的等级赋分模型。

used to dispatch score value in Zhejiang, Shanghai, Beijing, Tianjin, Shandong, New High Test projects。

等级分数转换通过以下步骤完成：
1. 根据规定比例，划分原始分数为各个等级区间
2. 每个等级区间赋予分数（山东方案使用线性转换为多个分值，其他省方案直接标定一个分数）
3. 各个考生的原始分数，先对应到等级，再转换为等级分数。

计算过程：
1. 计算分数字段的分段表，分段表包括四个列（字段）：分段值（分值点），对应该分段值的的人数，百分比，累积百分比，分段值按照顺序（默认从高到低）
2. 在分段表中计算，每个分段值对应的等级，等级分数
3. 使用分段表计算每个考生原始分数对应的等级分数

算法设计：
1. 精度问题
2. 区间端点计算方式



# 如何使用STM模块 how to use stm 

1. import module 导入模块
   
   [1] import pyex_stm as stm
   
    # 使用山东模型对数据框data中的分数字段ls和dl进行转换
    # use model shandong to transform at score field ls and dl
   
   [2] result = stm.run(name='shandong', df=data, field_list=('ls', 'dl'))
   --- start calculating map_table ---
    --start transform score field: <<ls>>
      use filter: [(df.ls>=0.0) & (df.ls<=96.0)]
   -- get input score endpoints ...
      begin calculating ...
      merge score field: ls_plt
      create report ...
   --start transform score field: <<dl>>
      use filter: [(df.dl>=0.0) & (df.dl<=96.0)]
   -- get input score endpoints ...
      begin calculating ...
      merge score field: dl_plt
      create report ...
   used time: 3.1701080799102783
   --------------------------------------------------
   
   [3] result.output_data.head()  # 查看运行结果 view the result
   Out[1]: 
      dl  ls  dl_plt  ls_plt
   0  85  90      94      99
   1  92  85      98      97
   2  92  73      98      90
   3  88  90      96      99
   4  93  76      98      92
      
2. get module information from function see() 调用see()函数查看模块的信息

   stm.help_doc（）
   
   module function and class:
    
   [function] 模块中的主要函数
   
          run(name, df, field_list, ratio, level_max, level_diff, input_score_max, input_score_min,
          
           output_score_decimal=0, approx_mode='near'): 运行各个模型的接口函数
           
          
          通过指定name=‘shandong'/'shanghai'/'zhejiang'/'beijing'/'tianjin'/'tao'
          
          可以计算山东、上海、浙江、北京、天津、陶百强模型
          
          通过指定name = 'zscore'/'tscore'/'tlinear'
          
          也可以计算Z分数、T分数、线性转换T分数
          
          ---
          
          interface function
          
          caculate shandong... model by name = 'shandong' / 'shanghai'/'zhejiang'/'beijing'/'tianjin'/'tao'
          
          caculate Z,T,liear T score by name = 'zscore'/ 'tscore' / 'tlinear'
          
          ---
          
          parameters specification:
          
          name: model name
          
          df: input raw score data, type DataFrame of pandas
          
          field_list: score field to calculate in df
          
          ratio: ratio list including percent value for each interval of level score
          
          level_max: max value of level score
          
          level_diff: differentiao value of level score
          
          input_score_max: raw score max value
          
          input_score_min: raw score min value
          
          output_score_decimal: level score precision, decimal digit number
          
          approx_mode: how to approxmate score points of raw score for each ratio vlaue
          
          ---
          
          usage:
          
          [1] import pyex_stm as stm
          
          [2] result = stm.run(name='shandong', df=data, field_list=['ls'])
          
          [3] result.report()
          
          [4] result.output.head()
          
       plot(): 
          
          各方案按照比例转换后分数后的分布直方图
          
          plot models distribution hist graph including shandong,zhejiang,shanghai,beijing,tianjin
          
       round45i():
          
          四舍五入函数
          
          function for rounding strictly at some decimal position
          
       get_norm__dist_table(size, mean, std, stdnum): 
          
          根据均值和标准差数生成正态分布表
       
          creating norm data dataframe with assigned mean and standard deviation
    
    [class] 模块中的类
    
       PltScore: 分段线性转换模型, 山东省新高考改革使用 shandong model

       LevelScore: 等级分数转换模型, 浙江、上海、天津、北京使用 zhejiang shanghai tianjin beijing model

       Zscore: Z分数转换模型 zscore model
       
       Tscore: T分数转换模型 tscore model
       
       Tlinear: T分数线性转换模型 tscore model by linear transform mode
       
       SegTable: 计算分段表模型 segment table model
       
       TaoScore: 陶百强等级分数模型（由陶百强在其论文中提出）Tao Baiqiang model
    

   
3. use interface function run() 使用模块的入口函数run()

   [1] result = stm.run(name='shandong', df=data, field_list=['ls'])  
   
   calculate level score by using shandong model, subject is 'ls' in input dataframe "data"
   
4. result of model 模型运行结果

   display report 显示运行报告
   
   [1] result.report()
   
   ---<< score field: [lsn] >>---
   
    input score  mean, std: 43.84, 18.01

    input score percentage: [0.03, 0.07, 0.16, 0.24, 0.24, 0.16, 0.07, 0.03]

    input score  endpoints: [(92, 74), (73, 67), (66, 58), (57, 46), (45, 32), (31, 20), (19, 8), (7, 0)]

    output score endpoints: [(100, 91), (90, 81), (80, 71), (70, 61), (60, 51), (50, 41), (40, 31), (30, 21)]

    transform formulas: 
    
         0.5*(x-74)+91

         1.5*(x-67)+81

         1.125*(x-58)+71

         0.818182*(x-46)+61

         0.692308*(x-32)+51

         0.818182*(x-20)+41

         0.818182*(x-8)+31

         1.285714*(x-0)+21

    ------------------------------------------------------------------------------------------

# 分段表模型 SegTable
  
  用于生成某些分数范围的每个分值点的指定科目得该分数的人数。
  
  输入数据： 一组考生的某些科目的成绩（pandas.DataFrame）。每个记录为一个考生，列指定考生考试科目。
  
  输出数据：seg(分段分数点），[字段名]_count, []_percent, []_cumsum

  应用SegTable.helpdoc()可以查看说明：
  
      * 计算pandas.DataFrame中分数字段的分段人数表
    * segment table for score dataframe
    * version1.01, 2018-06-21
    * version1.02, 2018-08-31
    * from 09-17-2017

    输入数据：分数表（pandas.DataFrame）,  计算分数分段人数的字段（list）
    set_data(input_data:DataFrame, field_list:list)
        input_data: input dataframe, with a value fields(int,float) to calculate segment table
                用于计算分段表的数据表，类型为pandas.DataFrmae
        field_list: list, field names used to calculate seg table, empty for calculate all fields
                   用于计算分段表的字段，多个字段以字符串列表方式设置，如：['sf1', 'sf2']
                   字段的类型应为可计算类型，如int,float.

    设置参数：最高分值，最低分值，分段距离，分段开始值，分数顺序，指定分段值列表， 使用指定分段列表，使用所有数据， 关闭计算过程显示信息
    set_parameters（segmax, segmin, segstep, segstart, segsort, seglist, useseglist, usealldata, display）
        segmax: int, maxvalue for segment, default=150
                输出分段表中分数段的最大值
        segmin: int, minvalue for segment, default=0。
                输出分段表中分数段的最小值
        segstep: int, grades for segment value, default=1
                分段间隔，用于生成n-分段表（五分一段的分段表）
        segstart:int, start seg score to count
                进行分段计算的起始值
        segsort: str, 'a' for ascending order or 'd' for descending order, default='d' (seg order on descending)
                输出结果中分段值得排序方式，d: 从大到小， a：从小到大
                排序模式的设置影响累计数和百分比的意义。
        seglist: list, used to create set value
                 使用给定的列表产生分段表，列表中为分段点值
        useseglist: bool, use or not use seglist to create seg value
                 是否使用给定列表产生分段值
        usealldata: bool, True: consider all score , the numbers outside are added to segmin or segmax
                 False: only consider score in [segmin, segmax] , abort the others records
                 default=False.
                 考虑最大和最小值之外的分数记录，高于的segmax的分数计数加入segmax分数段，
                 低于segmin分数值的计数加入segmin分数段
        display: bool, True: display run() message include time consume, False: close display message in run()
                  打开（True）或关闭（False）在运行分段统计过程中的显示信息
    output_data: 输出分段数据
            seg: seg value
        [field]: field name in field_list
        [field]_count: number at the seg
        [field]_sum: cumsum number at the seg
        [field]_percent: percentage at the seg
        [field]_count[step]: count field for step != 1
        [field]_list: count field for assigned seglist when use seglist
    运行，产生输出数据, calculate and create output data
    run()

    应用举例
    example:
        seg = SegTable()
        df = pd.DataFrame({'sf':[i % 11 for i in range(100)]})
        seg.set_data(df, ['sf'])
        seg.set_parameters(segmax=100, segmin=1, segstep=1, segsort='d', usealldata=True, display=True)
        seg.run()
        seg.plot()
        print(seg.output_data.head())    # get result dataframe, with fields: sf, sf_count, sf_cumsum, sf_percent

    Note:
        1)根据usealldata确定是否在设定的区间范围内计算分数值
          usealldata=True时抛弃不在范围内的记录项
          usealldata=False则将高于segmax的统计数加到segmax，低于segmin的统计数加到segmin
          segmax and segmin used to constrain score value scope to be processed in [segmin, segmax]
          segalldata is used to include or exclude data outside [segmin, segmax]

        2)分段字段的类型为整数或浮点数（实数）
          field_list type is digit, for example: int or float

        3)可以单独设置数据(input_data),字段列表（field_list),各项参数（segmax, segmin, segsort,segalldata, segmode)
          如，seg.field_list = ['score_1', 'score_2'];
              seg.segmax = 120
          重新设置后需要运行才能更新输出数据ouput_data, 即调用run()
          便于在计算期间调整模型。
          by usting property mode, rawdata, scorefields, parameters can be setted individually
        4) 当设置大于1分的分段分值X时， 会在结果DataFrame中生成一个字段[segfiled]_countX，改字段中不需要计算的分段
          值设为-1。
          when segstep > 1, will create field [segfield]_countX, X=str(segstep), no used value set to -1 in this field
    

  


# 基于BERT的中文期刊论文分类方法研究
## 本人本科毕业设计
使用数据集：https://github.com/ydli-ai/CSL  
本项目中仅保留了50条样本数据，原始数据集请自行下载。  
项目代码是在服务器上运行后，下载到本地整理的，因此部分代码中的路径可能需要修改。  
### 项目结构
```
├─dataset   //数据集文件夹
│  ├─data   //原始数据
│  │      cleanQuotingError.ipynb   //清洗引号错误
│  │      dataClean.ipynb   //数据清洗
│  │      
│  ├─fineTune   //微调数据集
│  │  │  analyseLength.ipynb  //分析数据集长度
│  │  │  classification_dict.py //类别编解码字典及统计信息
│  │  │  genBlancedFineTuneData.ipynb //生成类别平衡的微调数据集
│  │  └─ genFineTuneData.ipynb //生成微调数据集
│  │          
│  └─preTrain  //增量预训练数据集
│      │  genLongDataset.ipynb  //生成长文本数据集
│      │  genLongStn.ipynb //拼接得到长文本原始数据
│      │  genParagraph.ipynb //拆分得到原始段落段落数据
│      │  genParagraphDataset.ipynb //生成段落数据集
│      │  genShortDataset.ipynb //生成短文本数据集
│      │  genShortsStn.ipynb //拆分得到短文本原始数据
│      └─ get_chinese_ref.py //WWM工具函数
│              
├─exp1  //实验1 ： 基准实验
│      fineTune1.ipynb 
│      
├─exp2&3 //实验2&3 ： 层次递减学习率与长文本截断方法
│      fineTune2.ipynb
│      
├─exp4 //实验4 ： 后四层特征融合
│      BertLastFourLayer.py
│      fineTune4.ipynb
│       
├─exp5 //实验5 ： 改进型损失函数处理数据类别不平衡
│      BertWithCEAndDSCLoss.py
│      BertWithDSCLoss.py
│      BertWithFocalLoss.py
│      BertWithWeightCELoss.py
│      fineTune5.ipynb
│      
├─exp6 //实验6 ： 对抗训练
│      BertWithFGM.py
│      BertWithFGSM.py
│      BertWithPGD.py
│      FGSM_FGM_fineTune.ipynb
│      PGD_fineTune.ipynb
│      
└─exp7&8 //实验7&8 ： 增量预训练
        fineTune1.py
        preTrain.py
        task.py
```

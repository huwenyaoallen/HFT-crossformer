## 数据预处理
```
python preprocess.py --raw_folder ./full_dataset/home/ubuntu/quant/tick_data/final_lob_data/min30_all_labeled_sample/ --save_folder ./30min_datasets_full/data --save_timestamps_path ./30min_datasets_full/timestamps.npy
```
脚本将会自动处理--raw_folder下的所有parquet文件，处理后的csv保存在--save_folder，所有涉及到的timestamp保存在--save_timestamps_path下

## 训练
```
bash pred.sh
```
脚本将会自动训练，结果保存在results下的对应setting里，pred.csv和true.csv即各时刻log return的预测值和真值，可用于后续计算metric
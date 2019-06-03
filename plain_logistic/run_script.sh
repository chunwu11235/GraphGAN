nohup python3 train.py --model_dir=tmp1/ --bilinear=True --learning_rate=0.01 --dropout=0.3 > result/model_1.txt 2>&1
nohup python3 train.py --model_dir=tmp2/ --bilinear=True --learning_rate=0.01 > result/model_2.txt 2>&1
nohup python3 train.py --model_dir=tmp3/ --bilinear=True --learning_rate=0.001 > result/model_3.txt 2>&1
#nohup python3 train.py --model_dir=tmp4/ --learning_rate=0.01 > model_4.txt 2>&1
#nohup python3 train.py --model_dir=tmp5/ --learning_rate=0.001 > model_5.txt 2>&1




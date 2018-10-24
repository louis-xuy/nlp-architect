### 训练模型

```shell
PYTHONPATH=. python3.6 examples/word_language_model_with_lstm/train.py --embedding True --input_file datasets/gen_data/gongchengche_2018_10_08.csv --name gcc --learning_rate 0.005 --num_steps 26 --batch_size 32 --n_iterations 10000
python train.py  --input_file data/jay.txt --num_steps 20 --batch_size 32 --name jay --n_iterations 5000 --learning_rate 0.01 --n_layers 3 --embedding True
```
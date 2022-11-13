# An-Introduction-to-DL-for-the-Physical-
## An introduction to Deep learning for thr Physical layer

### default parameter
```
parser.add_argument('--n', type=int, default=2)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.00017, help='learning rate')
parser.add_argument('--snr_db',type=float,default=7,help='training snr dB')
parser.add_argument('--train_num_msgs',type=float,default=10000,help='training num msgs')
parser.add_argument('--test_num_msgs',type=float,default=10000,help='testing num msgs')
parser.add_argument('--test', type=int, default=0)      # 0: train / 1 :test
parser.add_argument('--plot', type=int, default=0)      # 0: constellation / 1: ber curve
parser.add_argument('--weights', type=str,default= 'MY_MODEL_M=4_n=2_TrainSNR=7.pth')
```
### train
```
python mymain.py --test 0
```

### test
```
python mymain.py --test 1 --weight '{parameter_name.pth}'
```

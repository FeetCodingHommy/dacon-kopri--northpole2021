import numpy as np
from tqdm import tqdm
from glob import glob
import os
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf
# import gc

from dacon_baseline.dacon_functions import train_map_func
from custom_model.baseline_convlstm_seq2seq import Encoder, Decoder, Seq2Seq
from dacon_baseline.dacon_functions import loss_function
from dacon_baseline.metrics import mae_score, f1_score, mae_over_f1
from dacon_baseline.dacon_functions import predict


# 데이터

default_path = '/data/DACON_NORTHPOLE/'

train = pd.read_csv(os.path.join(default_path,'weekly_train.csv'))
test = pd.read_csv(os.path.join(default_path,'public_weekly_test.csv'))

# 데이터 전처리
#     테스트 활용 가능 마지막 제공 데이터와 맞춰야하는 기간 사이에는 2주의 공백이 있습니다.
#     과거 12주의 해빙 변화를 보고 2주 뒤부터 12주간의 변화를 예측하는 모델을 만들겠습니다.

train_data_path = default_path + 'weekly_train/' + train.tail(52*30)['week_file_nm'].values

input_window_size = 12
target_window_size = 12
gap = 2
step = 1
input_data_list, target_data_list = [], []

for i in range(0, len(train_data_path)-input_window_size-target_window_size-gap+1, step):
    input_data = train_data_path[i:i+input_window_size]
    target_data = train_data_path[i+input_window_size+gap:i+input_window_size+gap+target_window_size]
    input_data_list.append(input_data)
    target_data_list.append(target_data)

# 데이터셋
#     학습과 검증용 데이터로 분리합니다.
#     최근 1년(52주)을 검증 데이터셋으로 사용하였습니다.

BATCH_SIZE = 2

train_dataset = tf.data.Dataset.from_tensor_slices((input_data_list[:-52], target_data_list[:-52]))
train_dataset = train_dataset.map(lambda item1, item2: tf.numpy_function(train_map_func, [item1, item2], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((input_data_list[-52:], target_data_list[-52:]))
val_dataset = val_dataset.map(lambda item1, item2: tf.numpy_function(train_map_func, [item1, item2], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

sample_enc_input_data = next(iter(train_dataset))[0]

# 모델

sample_encoder = Encoder(16, 1)
enc_output = sample_encoder(sample_enc_input_data)
# enc_output[0].shape, enc_output[1].shape

sample_decoder = Decoder(16)
dec_output = sample_decoder(enc_output)
# dec_output.shape

model = Seq2Seq(16, 1, 1)

# 학습률 & 옵티마이저

learning_rate = 0.0005
optimizer = tf.keras.optimizers.Adam(learning_rate)

# 체크포인트

checkpoint_path = './checkpoint/'
os.makedirs(checkpoint_path, exist_ok=True)
ckpt = tf.train.Checkpoint(
    Seq2Seq=model, 
    optimizer=optimizer
)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(ckpt_manager.latest_checkpoint)

# 학습

EPOCHS = 50

@tf.function
def train_step(inp, targ, training):
    loss = 0
    with tf.GradientTape() as tape:
        output = model(inp)
        for t in range(targ.shape[1]):
            loss += loss_function(targ[:, t], output[:, t])
            
    batch_loss = (loss / int(targ.shape[1]))
    
    if training==True:
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        
    return batch_loss

def val_score(inp, targ):
    output = model(inp)
    score = mae_over_f1(targ.numpy(), output.numpy())
    return score

loss_plot, val_score_plot = [], []
for epoch in range(EPOCHS):
    total_loss, total_val_score = 0, 0
    
    tqdm_dataset = tqdm(enumerate(train_dataset))
    for (batch, (inp, targ)) in tqdm_dataset:
        batch_loss = train_step(inp, targ, True)
        total_loss += batch_loss
        
        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Loss': '{:06f}'.format(250*batch_loss.numpy().mean()),
            'Total Loss' : '{:06f}'.format(250*total_loss/(batch+1))
        })
    loss_plot.append(250*total_loss/(batch+1))
    
    tqdm_dataset_val = tqdm(enumerate(val_dataset))
    for (batch, (inp, targ)) in tqdm_dataset_val:
        batch_val_score = val_score(inp, targ)
        total_val_score += batch_val_score.mean()
        
        tqdm_dataset_val.set_postfix({
            'Epoch': epoch + 1,
            'Val Score': '{:06f}'.format(250*batch_val_score.mean()),
            'Val Total_Score' : '{:06f}'.format(250*total_val_score/(batch+1))
        })
    val_score_plot.append(250*total_val_score/(batch+1))
    
    if np.min(val_score_plot) == val_score_plot[-1]:
        ckpt_manager.save()

# 학습 결과

plt.plot(loss_plot)
plt.title('loss_plot')
plt.show()

plt.plot(val_score_plot)
plt.title('val_score_plot')
plt.show()

# 모델복원

ckpt.restore(ckpt_manager.latest_checkpoint)

# 추론

test_path = default_path + 'weekly_train/' + test.tail(12)['week_file_nm']

pred = predict(test_path)

# 제출

submission = pd.read_csv('./sample_submission.csv')

sub_2020 = submission.loc[:11, ['week_start']].copy()
sub_2021 = submission.loc[12:].copy()

sub_2020 = pd.concat([sub_2020, (pd.DataFrame(pred.reshape([12,-1])))], axis=1)
sub_2021.columns = sub_2020.columns
submission = pd.concat([sub_2020, sub_2021])

submission.to_csv('./baseline.csv', index=False)

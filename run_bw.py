from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf
# import gc

from custom_model.s2s_lstm2lstm import Encoder, Decoder
from utils.dacon_functions import predict
from utils.my_metrics import mae_score
from utils.my_utils import DataGenerator, my_cycle_scheduler


# 학과 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the second GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except Exception as e:
        print(e)
        raise KeyboardInterrupt

# 데이터

default_path = '/data/hommy/arctic_data/'

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

BATCH_SIZE = 1

train_data_gen = DataGenerator(input_data_list[:-52], target_data_list[:-52], batch_size=BATCH_SIZE)
valid_data_gen = DataGenerator(input_data_list[-52:], target_data_list[-52:], batch_size=BATCH_SIZE)

# 모델

image_height = 448
image_width = 304
image_channel = 1
hidden_dim = 16

# Define an input sequence and process it.
encoder_inputs = tf.keras.Input(shape=(input_window_size, image_height, image_width, image_channel), batch_size=BATCH_SIZE)
encoder_inputs_reversed = tf.reverse(encoder_inputs, axis=[1], name="input_reverse")

encoder = Encoder(hidden_dim, 1)
enc_output = encoder(encoder_inputs)
# enc_output[0].shape, enc_output[1].shape

decoder = Decoder(hidden_dim)
dec_output = decoder(enc_output)
decoder_inputs_reversed = tf.reverse(dec_output, axis=[1], name="output_reverse")
# dec_output.shape

# model = Seq2Seq(16, 1, 1)
model = tf.keras.Model(encoder_inputs, dec_output)

# 학습률 & 옵티마이저

learning_rate = 0.0005      # 0.0005(baseline)~0.00005~0.000005
# optimizer = tf.keras.optimizers.Adam(learning_rate)
learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(my_cycle_scheduler)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.MAE,
    metrics=[mae_score] # [mae_score, f1_score, mae_over_f1]
)

# 체크포인트

checkpoint_path = './checkpoint_tanh_bw/'
os.makedirs(checkpoint_path, exist_ok=True)

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss')

# 학습

EPOCHS = 200

history = model.fit(train_data_gen, validation_data=valid_data_gen,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=[learning_rate_scheduler, model_checkpoint_callback],
                    verbose=True)

# 학습 결과

plt.plot(history.history["loss"])
plt.title('loss_plot')
plt.savefig("./elmo_tanh_bw_loss_plot.png")
plt.clf()

plt.plot(history.history["val_loss"])
plt.title('val_loss_plot')
plt.savefig("./elmo_tanh_bw_val_loss_plot.png")
plt.clf()

# 모델복원

model.load_weights(checkpoint_path)

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

submission.to_csv('./elmo_backward_model.csv', index=False)

print("WOW!")

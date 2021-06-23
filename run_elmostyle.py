from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf
# import gc

from custom_model.s2s_lstm2lstm import Encoder, Decoder
from custom_model.weighted_average import WeightMultiply
from utils.dacon_functions import predict
from utils.my_metrics import mae_score
from utils.my_utils import DataGenerator


# 학과 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the last GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
    except Exception as e:
        print(e)
        raise KeyboardInterrupt

# 데이터

default_path = '/data/hommy/arctic_data_v2/'

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

encoder1 = Encoder(hidden_dim, 1)
enc_output1 = encoder1(encoder_inputs)
# enc_output[0].shape, enc_output[1].shape

decoder1 = Decoder(hidden_dim)
dec_output1 = decoder1(enc_output1)
# dec_output.shape

model1 = tf.keras.Model(encoder_inputs, dec_output1)
model1.load_weights("./checkpoint_tanh_fw/")

# Define an input sequence and process it.
encoder_inputs_reversed = tf.reverse(encoder_inputs, axis=[1], name="input_reverse")

encoder2 = Encoder(hidden_dim, 1)
enc_output2 = encoder2(encoder_inputs_reversed)
# enc_output[0].shape, enc_output[1].shape

decoder2 = Decoder(hidden_dim)
dec_output2 = decoder2(enc_output2)
decoder_inputs_reversed = tf.reverse(dec_output2, axis=[1], name="output_reverse")
# dec_output.shape

model2 = tf.keras.Model(encoder_inputs, decoder_inputs_reversed)
model2.load_weights("./checkpoint_tanh_bw/")

for l in model1.layers:
    l.trainable = False

for l in model2.layers:
    l.trainable = False

model1_outputs = tf.split(model1.outputs, num_or_size_splits=12, axis=-4)
model2_outputs = tf.split(model2.outputs, num_or_size_splits=12, axis=-4)

outputs_combined = list()
for i, (o1, o2) in enumerate(zip(model1_outputs, model2_outputs)):
    weights = WeightMultiply(
        weight_shape=(BATCH_SIZE, image_height, image_width, image_channel), 
        trainable=True, 
        name=f"weights_{i+1}"
    )
    w_o1, w_o2 = weights(tf.squeeze(o1, axis=0), tf.squeeze(o2, axis=0))
    o_c = tf.keras.layers.Add()([w_o1, w_o2])
    outputs_combined.append(o_c)

outputs_tensor = tf.concat(outputs_combined, axis=-4)

model = tf.keras.Model(model1.input, outputs_tensor)

# 학습률 & 옵티마이저

max_learning_rate = 0.02
learning_rate = 0.002

def scheduler(epoch, lr):
  if epoch < 30:
    return (max_learning_rate - lr) * (31-epoch) / 30 + lr
  else:
    return lr

learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.MAE,
    metrics=[mae_score] # [mae_score, f1_score, mae_over_f1]
)

# 체크포인트

checkpoint_path = './checkpoint_tanh_precomputed/'
os.makedirs(checkpoint_path, exist_ok=True)

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss')

# 학습

EPOCHS = 50

history = model.fit(train_data_gen, validation_data=valid_data_gen,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=[learning_rate_scheduler, model_checkpoint_callback],
                    verbose=True)

# 학습 결과

plt.plot(history.history["loss"])
plt.title('loss_plot')
plt.savefig("./elmo_tanh_loss_plot.png")
plt.clf()

plt.plot(history.history["val_loss"])
plt.title('val_loss_plot')
plt.savefig("./elmo_tanh_val_loss_plot.png")
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

submission.to_csv('./elmo_precomputed.csv', index=False)

print("WOW!")

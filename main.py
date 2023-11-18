# %% [markdown]
# 18-layers small (192 dim) transformer structure.        
# AdamW optimizer.  
# Batch size 64.  
# 800 epochs.  
# [Inference notebook](https://www.kaggle.com/code/shlomoron/srrf-transformer-tpu-inference)

# %% [markdown]
# # Imports

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:42:09.791670Z","iopub.execute_input":"2023-10-31T20:42:09.792013Z","iopub.status.idle":"2023-10-31T20:42:51.394633Z","shell.execute_reply.started":"2023-10-31T20:42:09.791985Z","shell.execute_reply":"2023-10-31T20:42:51.393753Z"}}
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shutil
import math
import gc
import os

# %% [markdown]
# # TPU boilerplate code

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:42:51.396278Z","iopub.execute_input":"2023-10-31T20:42:51.396737Z","iopub.status.idle":"2023-10-31T20:42:59.547535Z","shell.execute_reply.started":"2023-10-31T20:42:51.396709Z","shell.execute_reply":"2023-10-31T20:42:59.546424Z"}}
# Configure Strategy. Assume TPU...if not set default for GPU
tpu = None
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local") # "local" for 1VM TPU
    strategy = tf.distribute.TPUStrategy(tpu)
    print("on TPU")
    print("REPLICAS: ", strategy.num_replicas_in_sync)
except:
    strategy = tf.distribute.get_strategy()

# %% [markdown]
# # Configs

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:42:59.548803Z","iopub.execute_input":"2023-10-31T20:42:59.549142Z","iopub.status.idle":"2023-10-31T20:42:59.554482Z","shell.execute_reply.started":"2023-10-31T20:42:59.549108Z","shell.execute_reply":"2023-10-31T20:42:59.553651Z"}}
DEBUG = False

PAD_x = 0.0
PAD_y = np.nan
X_max_len = 206
batch_size = 64

if DEBUG:
    batch_size = 2

num_vocab = 5
hidden_dim = 192

# %% [markdown]
# # Data API pipeline
# This section applies filtering, preprocessing, shuffling, paddings, and batchings. I already transformed all the data to TFRecords; you can find the TFRecords dataset [here](https://www.kaggle.com/datasets/shlomoron/srrf-tfrecords-ds). I shuffled the samples before creating the TFRecords.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:42:59.557860Z","iopub.execute_input":"2023-10-31T20:42:59.558228Z","iopub.status.idle":"2023-10-31T20:42:59.568322Z","shell.execute_reply.started":"2023-10-31T20:42:59.558197Z","shell.execute_reply":"2023-10-31T20:42:59.567421Z"}}
tffiles_path = "/kaggle/input/d/malazimad/srrf-tfrecords-ds/tfds"
tffiles = [f'{tffiles_path}/{x}.tfrecord' for x in range(164)]

# %% [markdown]
# ## Decoding TFRecords

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:42:59.569960Z","iopub.execute_input":"2023-10-31T20:42:59.570449Z","iopub.status.idle":"2023-10-31T20:42:59.589932Z","shell.execute_reply.started":"2023-10-31T20:42:59.570412Z","shell.execute_reply":"2023-10-31T20:42:59.589132Z"}}
def decode_tfrec(record_bytes):
    schema = {}
    schema["id"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["seq"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["dataset_name_2A3"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["dataset_name_DMS"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["reads_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reads_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["signal_to_noise_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["signal_to_noise_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["SN_filter_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["SN_filter_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reactivity_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reactivity_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reactivity_error_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reactivity_error_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    features = tf.io.parse_single_example(record_bytes, schema)

    sample_id = tf.sparse.to_dense(features["id"])
    seq = tf.sparse.to_dense(features["seq"])
    dataset_name_2A3 = tf.sparse.to_dense(features["dataset_name_2A3"])
    dataset_name_DMS = tf.sparse.to_dense(features["dataset_name_DMS"])
    reads_2A3 = tf.sparse.to_dense(features["reads_2A3"])
    reads_DMS = tf.sparse.to_dense(features["reads_DMS"])
    signal_to_noise_2A3 = tf.sparse.to_dense(features["signal_to_noise_2A3"])
    signal_to_noise_DMS = tf.sparse.to_dense(features["signal_to_noise_DMS"])
    SN_filter_2A3 = tf.sparse.to_dense(features["SN_filter_2A3"])
    SN_filter_DMS = tf.sparse.to_dense(features["SN_filter_DMS"])
    reactivity_2A3 = tf.sparse.to_dense(features["reactivity_2A3"])
    reactivity_DMS = tf.sparse.to_dense(features["reactivity_DMS"])
    reactivity_error_2A3 = tf.sparse.to_dense(features["reactivity_error_2A3"])
    reactivity_error_DMS = tf.sparse.to_dense(features["reactivity_error_DMS"])

    out = {}
    out['seq']  = seq
    out['SN_filter_2A3']  = SN_filter_2A3
    out['SN_filter_DMS']  = SN_filter_DMS
    out['reads_2A3']  = reads_2A3
    out['reads_DMS']  = reads_DMS
    out['signal_to_noise_2A3']  = signal_to_noise_2A3
    out['signal_to_noise_DMS']  = signal_to_noise_DMS
    out['reactivity_2A3']  = reactivity_2A3
    out['reactivity_DMS']  = reactivity_DMS
    return out

# %% [markdown]
# ## Filtering

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:42:59.591281Z","iopub.execute_input":"2023-10-31T20:42:59.591694Z","iopub.status.idle":"2023-10-31T20:42:59.603804Z","shell.execute_reply.started":"2023-10-31T20:42:59.591659Z","shell.execute_reply":"2023-10-31T20:42:59.602937Z"}}
def f1(): return True
def f2(): return False

def filter_function_1(x):
    SN_filter_2A3 = x['SN_filter_2A3']
    SN_filter_DMS = x['SN_filter_DMS']
    return tf.cond((SN_filter_2A3 == 1) and (SN_filter_DMS == 1) , true_fn=f1, false_fn=f2)

def filter_function_2(x):
    reads_2A3 = x['reads_2A3']
    reads_DMS = x['reads_DMS']
    signal_to_noise_2A3 = x['signal_to_noise_2A3']
    signal_to_noise_DMS = x['signal_to_noise_DMS']
    cond = (reads_2A3>100 and signal_to_noise_2A3>0.75) or (reads_DMS>100 and signal_to_noise_DMS>0.75)
    return tf.cond(cond, true_fn=f1, false_fn=f2)

# %% [markdown]
# ## Preprocessing

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:42:59.605176Z","iopub.execute_input":"2023-10-31T20:42:59.605880Z","iopub.status.idle":"2023-10-31T20:42:59.615198Z","shell.execute_reply.started":"2023-10-31T20:42:59.605841Z","shell.execute_reply":"2023-10-31T20:42:59.614288Z"}}
def nan_below_filter(x):
    reads_2A3 = x['reads_2A3']
    reads_DMS = x['reads_DMS']
    signal_to_noise_2A3 = x['signal_to_noise_2A3']
    signal_to_noise_DMS = x['signal_to_noise_DMS']
    reactivity_2A3 = x['reactivity_2A3']
    reactivity_DMS = x['reactivity_DMS']

    if reads_2A3<100 or signal_to_noise_2A3<0.75:
        reactivity_2A3 = np.nan+reactivity_2A3
    if reads_DMS<100 or signal_to_noise_DMS<0.75:
        reactivity_DMS = np.nan+reactivity_DMS

    x['reactivity_2A3'] = reactivity_2A3
    x['reactivity_DMS'] = reactivity_DMS
    return x

def concat_target(x):
    reactivity_2A3 = x['reactivity_2A3']
    reactivity_DMS = x['reactivity_DMS']
    target = tf.concat([reactivity_2A3[..., tf.newaxis], reactivity_DMS[..., tf.newaxis]], axis = 1)
    target = tf.clip_by_value(target, 0, 1)
    return x['seq'], target

# %% [markdown]
# ## get_tfrec_dataset

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:42:59.616569Z","iopub.execute_input":"2023-10-31T20:42:59.616959Z","iopub.status.idle":"2023-10-31T20:42:59.629686Z","shell.execute_reply.started":"2023-10-31T20:42:59.616923Z","shell.execute_reply":"2023-10-31T20:42:59.628918Z"}}
def get_tfrec_dataset(tffiles, shuffle, batch_size, cache = False, to_filter = False,
                      calculate_sample_num = True):
    ds = tf.data.TFRecordDataset(
        tffiles, num_parallel_reads=tf.data.AUTOTUNE, compression_type = 'GZIP').prefetch(tf.data.AUTOTUNE)

    ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    if to_filter == 'filter_1':
        ds = ds.filter(filter_function_1)
    elif to_filter == 'filter_2':
        ds = ds.filter(filter_function_2)
    ds = ds.map(nan_below_filter, tf.data.AUTOTUNE)
    ds = ds.map(concat_target, tf.data.AUTOTUNE)

    if DEBUG:
        ds = ds.take(8)

    if cache:
        ds = ds.cache()

    samples_num = 0
    if calculate_sample_num:
        samples_num = ds.reduce(0, lambda x,_: x+1).numpy()

    if shuffle:
        if shuffle == -1:
            ds = ds.shuffle(samples_num, reshuffle_each_iteration = True)
        else:
            ds = ds.shuffle(shuffle, reshuffle_each_iteration = True)

    if batch_size:
        ds = ds.padded_batch(
            batch_size, padding_values=(PAD_x, PAD_y), padded_shapes=([X_max_len],[X_max_len, 2]), drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, samples_num

# %% [markdown]
# ## Define datasets

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:42:59.630716Z","iopub.execute_input":"2023-10-31T20:42:59.631075Z","iopub.status.idle":"2023-10-31T20:42:59.640918Z","shell.execute_reply.started":"2023-10-31T20:42:59.631039Z","shell.execute_reply":"2023-10-31T20:42:59.640214Z"}}
val_len = 5
if DEBUG:
    val_len = 1

val_files = tffiles[:val_len]

if DEBUG:
    train_files = tffiles[val_len:val_len+1]
else:
    train_files = tffiles[val_len:]

# %% [markdown]
# ## Get datasets

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:42:59.645257Z","iopub.execute_input":"2023-10-31T20:42:59.645545Z","iopub.status.idle":"2023-10-31T20:43:58.929318Z","shell.execute_reply.started":"2023-10-31T20:42:59.645521Z","shell.execute_reply":"2023-10-31T20:43:58.928205Z"}}
train_dataset, num_train = get_tfrec_dataset(train_files, shuffle = -1, batch_size = batch_size,
                                                  cache = True, to_filter = 'filter_2', calculate_sample_num = True)

val_dataset, num_val = get_tfrec_dataset(val_files, shuffle = False, batch_size = batch_size,
                                                  cache = True, to_filter = 'filter_1', calculate_sample_num = True)
print(num_train)
print(num_val)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:43:58.930486Z","iopub.execute_input":"2023-10-31T20:43:58.930772Z","iopub.status.idle":"2023-10-31T20:43:58.982584Z","shell.execute_reply.started":"2023-10-31T20:43:58.930747Z","shell.execute_reply":"2023-10-31T20:43:58.981704Z"}}
batch = next(iter(val_dataset))
batch[0].shape, batch[1].shape

# %% [markdown]
# # Model

# %% [markdown]
# ## Model layers

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:43:58.983675Z","iopub.execute_input":"2023-10-31T20:43:58.983942Z","iopub.status.idle":"2023-10-31T20:43:58.998470Z","shell.execute_reply.started":"2023-10-31T20:43:58.983919Z","shell.execute_reply":"2023-10-31T20:43:58.997634Z"}}
class transformer_block(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim//num_heads)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(feed_forward_dim, activation="relu"),
                tf.keras.layers.Dense(dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.supports_masking = True

    def call(self, inputs, training, mask):
        att_mask = tf.expand_dims(mask, axis=-1)
        att_mask = tf.repeat(att_mask, repeats=tf.shape(att_mask)[1], axis=-1)

        attn_output = self.att(inputs, inputs, attention_mask = att_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class positional_encoding_layer(tf.keras.layers.Layer):
    def __init__(self, num_vocab=5, maxlen=500, hidden_dim=384):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_emb = self.positional_encoding(maxlen-1, hidden_dim)
        self.supports_masking = True

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        x = tf.math.multiply(x, tf.math.sqrt(tf.cast(self.hidden_dim, tf.float32)))
        return x + self.pos_emb[:maxlen, :]

    def positional_encoding(self, maxlen, hidden_dim):
        depth = hidden_dim/2
        positions = tf.range(maxlen, dtype = tf.float32)[..., tf.newaxis]
        depths = tf.range(depth, dtype = tf.float32)[np.newaxis, :]/depth
        angle_rates = tf.math.divide(1, tf.math.pow(tf.cast(10000, tf.float32), depths))
        angle_rads = tf.linalg.matmul(positions, angle_rates)
        pos_encoding = tf.concat(
          [tf.math.sin(angle_rads), tf.math.cos(angle_rads)],
          axis=-1)
        return pos_encoding

# %% [markdown]
# ## Loss function

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:43:58.999431Z","iopub.execute_input":"2023-10-31T20:43:58.999690Z","iopub.status.idle":"2023-10-31T20:43:59.009428Z","shell.execute_reply.started":"2023-10-31T20:43:58.999662Z","shell.execute_reply":"2023-10-31T20:43:59.008746Z"}}
def loss_fn(labels, targets):
    labels_mask = tf.math.is_nan(labels)
    labels = tf.where(labels_mask, tf.zeros_like(labels), labels)
    mask_count = tf.math.reduce_sum(tf.where(labels_mask, tf.zeros_like(labels), tf.ones_like(labels)))
    loss = tf.math.abs(labels - targets)
    loss = tf.where(labels_mask, tf.zeros_like(loss), loss)
    loss = tf.math.reduce_sum(loss)/mask_count
    return loss

# %% [markdown]
# ## Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:43:59.010389Z","iopub.execute_input":"2023-10-31T20:43:59.010636Z","iopub.status.idle":"2023-10-31T20:43:59.024047Z","shell.execute_reply.started":"2023-10-31T20:43:59.010615Z","shell.execute_reply":"2023-10-31T20:43:59.023365Z"}}
def get_model(hidden_dim = 384, max_len = 206):
    with strategy.scope():
        inp = tf.keras.Input([max_len])
        x = inp

        x = tf.keras.layers.Embedding(num_vocab, hidden_dim, mask_zero=True)(x)
        x = positional_encoding_layer(num_vocab=num_vocab, maxlen=500, hidden_dim=hidden_dim)(x)

        x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
        x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
        x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
        x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)

        x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
        x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
        
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(2)(x)

        model = tf.keras.Model(inp, x)
        loss = loss_fn
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005)
        model.compile(loss=loss, optimizer=optimizer, jit_compile = True)
        return model

tf.keras.backend.clear_session()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:43:59.025153Z","iopub.execute_input":"2023-10-31T20:43:59.025390Z","iopub.status.idle":"2023-10-31T20:44:11.235358Z","shell.execute_reply.started":"2023-10-31T20:43:59.025370Z","shell.execute_reply":"2023-10-31T20:44:11.234120Z"}}
model = get_model(hidden_dim = 192,max_len = X_max_len)
model(batch[0])
model.summary()

# %% [markdown]
# ## Learning rate scheduler
# I copied the scheduler from [here](https://www.kaggle.com/code/irohith/aslfr-ctc-based-on-prev-comp-1st-place).

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:44:11.236526Z","iopub.execute_input":"2023-10-31T20:44:11.236784Z","iopub.status.idle":"2023-10-31T20:44:11.241190Z","shell.execute_reply.started":"2023-10-31T20:44:11.236759Z","shell.execute_reply":"2023-10-31T20:44:11.240332Z"}}
N_EPOCHS = 498
initial_epoch = 324
if DEBUG:
    N_EPOCHS = 5
N_WARMUP_EPOCHS = 0
LR_MAX = 5e-4
WD_RATIO = 0.05
WARMUP_METHOD = "exp"

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:44:11.242556Z","iopub.execute_input":"2023-10-31T20:44:11.242834Z","iopub.status.idle":"2023-10-31T20:44:13.282090Z","shell.execute_reply.started":"2023-10-31T20:44:11.242814Z","shell.execute_reply":"2023-10-31T20:44:13.280916Z"}}
def lrfn(current_step, num_warmup_steps, lr_max, num_cycles=0.50, num_training_steps=N_EPOCHS):
    if current_step < num_warmup_steps:
        if WARMUP_METHOD == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max

def plot_lr_schedule(lr_schedule, epochs):
    fig = plt.figure(figsize=(20, 10))
    plt.plot([None] + lr_schedule + [None])
    # X Labels
    x = np.arange(1, epochs + 1)
    x_axis_labels = [i if epochs <= 40 or i % 5 == 0 or i == 1 else None for i in range(1, epochs + 1)]
    plt.xlim([1, epochs])
    plt.xticks(x, x_axis_labels) # set tick step to 1 and let x axis start at 1

    # Increase y-limit for better readability
    plt.ylim([0, max(lr_schedule) * 1.1])

    # Title
    schedule_info = f'start: {lr_schedule[0]:.1E}, max: {max(lr_schedule):.1E}, final: {lr_schedule[-1]:.1E}'
    plt.title(f'Step Learning Rate Schedule, {schedule_info}', size=18, pad=12)

    # Plot Learning Rates
    for x, val in enumerate(lr_schedule):
        if epochs <= 40 or x % 5 == 0 or x is epochs - 1:
            if x < len(lr_schedule) - 1:
                if lr_schedule[x - 1] < val:
                    ha = 'right'
                else:
                    ha = 'left'
            elif x == 0:
                ha = 'right'
            else:
                ha = 'left'
            plt.plot(x + 1, val, 'o', color='black');
            offset_y = (max(lr_schedule) - min(lr_schedule)) * 0.02
            plt.annotate(f'{val:.1E}', xy=(x + 1, val + offset_y), size=12, ha=ha)

    plt.xlabel('Epoch', size=16, labelpad=5)
    plt.ylabel('Learning Rate', size=16, labelpad=5)
    plt.grid()
    plt.show()

# Learning rate for encoder
LR_SCHEDULE = [lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_cycles=0.50) for step in range(N_EPOCHS)]
# Plot Learning Rate Schedule
plot_lr_schedule(LR_SCHEDULE, epochs=N_EPOCHS)
# Learning Rate Callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=0)

# %% [markdown]
# ## Saving callback

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:44:13.283414Z","iopub.execute_input":"2023-10-31T20:44:13.283679Z","iopub.status.idle":"2023-10-31T20:44:13.289526Z","shell.execute_reply.started":"2023-10-31T20:44:13.283657Z","shell.execute_reply":"2023-10-31T20:44:13.288644Z"}}
save_folder = '/kaggle/working'
try:
    os.mkdir(f'{save_folder}/weights')
except:
    pass

class save_model_callback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    def on_epoch_end(self, epoch: int, logs=None):
        if epoch == 3 or (epoch+1)%25 == 0:
            self.model.save_weights(f"{save_folder}/weights/model_epoch_{epoch}.h5")

# %% [markdown]
# ## Training

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:44:13.290578Z","iopub.execute_input":"2023-10-31T20:44:13.291113Z","iopub.status.idle":"2023-10-31T20:49:03.391092Z","shell.execute_reply.started":"2023-10-31T20:44:13.291086Z","shell.execute_reply":"2023-10-31T20:49:03.389500Z"}}
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    initial_epoch=initial_epoch,
    epochs=N_EPOCHS,
    verbose = 2,
    callbacks=[
        save_model_callback(),
        lr_callback, 
    ]
)

# %% [markdown]
# ## Plotting loss

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-10-31T20:49:03.392047Z","iopub.status.idle":"2023-10-31T20:49:03.392435Z","shell.execute_reply.started":"2023-10-31T20:49:03.392236Z","shell.execute_reply":"2023-10-31T20:49:03.392253Z"}}
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# %% [code]
model.save_weights(f"/kaggle/working/weights/model_epoch_498.h5")

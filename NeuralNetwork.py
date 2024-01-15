import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn import preprocessing
from scipy import stats
import os #tensorflow complier flags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #tensoflow complier flags
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
from IPython import display

# Read csv file
df = pd.read_csv('/Users/mengwenwu/PycharmProjects/Stage22/PTO.csv')
df.head()
#    MONTH   HOUR  ... COMPLETENESS(%) Grade
# 0     10  19:00  ...           97.01     A
# 1     10  19:00  ...           97.22     A
# 2     12  21:00  ...           97.32     A
# 3      4  20:00  ...           97.17     A
# 4      9  20:00  ...           96.57     C
# [5 rows x 20 columns]

df.columns
# Index(['MONTH', 'HOUR', 'CITY - VENUE', 'HOME', 'AWAY', 'COMPETITION', 'ROUND',
#        'SYSTEM N°', 'OPERATOR', 'STABILIZATION RESS. NEEDS',
#        'WEATHER CONDITIONS', 'STADIUM CONDITIONS', 'INTERNET CONDITIONS',
#        'ISSUE TYPE', 'LIVE TRACKING ISSUE TYPE', 'POST  ISSUE TYPE',
#        'NO ISSUES', 'ISSUES COUNT', 'COMPLETENESS(%)', 'Grade'],
#       dtype='object')
df = df.drop(['NO ISSUES', 'COMPLETENESS(%)'], axis=1)

# Some formatting
# Convert integers to strings
df['MONTH'] = df['MONTH'].astype('string')
# Standardize Issues Count
df['ISSUES COUNT'] = stats.zscore(df['ISSUES COUNT'])
# Encode target category
le = preprocessing.LabelEncoder()
le.fit(df['Grade'].values)
le.classes_
labels_enc = le.transform(df['Grade'].values)
labels = tf.keras.utils.to_categorical(labels_enc)
labels

# Split data into train and test sets
df_train, df_test = train_test_split(df, random_state=0, train_size = .75)
df_train, df_test

# Numerical feature
issues_count = df_train['ISSUES COUNT']
issues_count = df_train[['ISSUES COUNT']].values
issues_count

# MONTH
docs1 = df_train['MONTH'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs1)
vocab_size1 = len(t.word_index) + 1
# Label encode the documents
encoded_docs1 = t.texts_to_sequences(docs1)
# Pad documents to 1
max_length = 1
padded_docs1 = pad_sequences(encoded_docs1, maxlen=max_length, padding='post')
vocab_size1

# HOUR
docs2 = df_train['HOUR'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs2)
vocab_size2 = len(t.word_index) + 1
# Label encode the documents
encoded_docs2 = t.texts_to_sequences(docs2)
# Pad documents to 1
max_length = 1
padded_docs2 = pad_sequences(encoded_docs2, maxlen=max_length, padding='post')
vocab_size2

# CITY - VENUE
docs3 = df_train['CITY - VENUE'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs3)
vocab_size3 = len(t.word_index) + 1
# Label encode the documents
encoded_docs3 = t.texts_to_sequences(docs3)
# Pad documents to 1
max_length = 1
padded_docs3 = pad_sequences(encoded_docs3, maxlen=max_length, padding='post')
vocab_size3

# HOME
docs4 = df_train['HOME'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs4)
vocab_size4 = len(t.word_index) + 1
# Label encode the documents
encoded_docs4 = t.texts_to_sequences(docs4)
# Pad documents to 1
max_length = 1
padded_docs4 = pad_sequences(encoded_docs4, maxlen=max_length, padding='post')
vocab_size4

# AWAY
docs5 = df_train['AWAY'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs5)
vocab_size5 = len(t.word_index) + 1
# Label encode the documents
encoded_docs5 = t.texts_to_sequences(docs5)
# Pad documents to 1
max_length = 1
padded_docs5 = pad_sequences(encoded_docs5, maxlen=max_length, padding='post')
vocab_size5

# COMPETITION
docs6 = df_train['COMPETITION'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs6)
vocab_size6 = len(t.word_index) + 1
# Label encode the documents
encoded_docs6 = t.texts_to_sequences(docs6)
# Pad documents to 1
max_length = 1
padded_docs6 = pad_sequences(encoded_docs6, maxlen=max_length, padding='post')
vocab_size6

# ROUND
docs7 = df_train['ROUND'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs7)
vocab_size7 = len(t.word_index) + 1
# Label encode the documents
encoded_docs7 = t.texts_to_sequences(docs7)
# Pad documents to 1
max_length = 1
padded_docs7 = pad_sequences(encoded_docs7, maxlen=max_length, padding='post')
vocab_size7

# SYSTEM N°
docs8 = df_train['SYSTEM N°'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs8)
vocab_size8 = len(t.word_index) + 1
# Label encode the documents
encoded_docs8 = t.texts_to_sequences(docs8)
# Pad documents to 1
max_length = 1
padded_docs8 = pad_sequences(encoded_docs8, maxlen=max_length, padding='post')
vocab_size8

# OPERATOR
docs9 = df_train['OPERATOR'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs9)
vocab_size9 = len(t.word_index) + 1
# Label encode the documents
encoded_docs9 = t.texts_to_sequences(docs9)
# Pad documents to 1
max_length = 1
padded_docs9 = pad_sequences(encoded_docs9, maxlen=max_length, padding='post')
vocab_size9

# STABILIZATION RESS. NEEDS
docs10 = df_train['STABILIZATION RESS. NEEDS'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs10)
vocab_size10 = len(t.word_index) + 1
# Label encode the documents
encoded_docs10 = t.texts_to_sequences(docs10)
# Pad documents to 1
max_length = 1
padded_docs10 = pad_sequences(encoded_docs10, maxlen=max_length, padding='post')
vocab_size10

# WEATHER CONDITIONS
docs11 = df_train['WEATHER CONDITIONS'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs11)
vocab_size11 = len(t.word_index) + 1
# Label encode the documents
encoded_docs11 = t.texts_to_sequences(docs11)
# Pad documents to 1
max_length = 1
padded_docs11 = pad_sequences(encoded_docs11, maxlen=max_length, padding='post')
vocab_size11

# STADIUM CONDITIONS
docs12 = df_train['STADIUM CONDITIONS'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs12)
vocab_size12 = len(t.word_index) + 1
# Label encode the documents
encoded_docs12 = t.texts_to_sequences(docs12)
# Pad documents to 1
max_length = 1
padded_docs12 = pad_sequences(encoded_docs12, maxlen=max_length, padding='post')
vocab_size12

# INTERNET CONDITIONS
df_train['INTERNET CONDITIONS']=df_train['INTERNET CONDITIONS'].astype('str')
docs13 = df_train['INTERNET CONDITIONS'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs13)
vocab_size13 = len(t.word_index) + 1
# Label encode the documents
encoded_docs13 = t.texts_to_sequences(docs13)
# Pad documents to 1
max_length = 1
padded_docs13 = pad_sequences(encoded_docs13, maxlen=max_length, padding='post')
vocab_size13

# ISSUE TYPE
docs14 = df_train['ISSUE TYPE'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs14)
vocab_size14 = len(t.word_index) + 1
# Label encode the documents
encoded_docs14 = t.texts_to_sequences(docs14)
# Pad documents to 1
max_length = 1
padded_docs14 = pad_sequences(encoded_docs14, maxlen=max_length, padding='post')
vocab_size14

# LIVE TRACKING ISSUE TYPE
docs15 = df_train['LIVE TRACKING ISSUE TYPE'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs15)
vocab_size15 = len(t.word_index) + 1
# Label encode the documents
encoded_docs15 = t.texts_to_sequences(docs15)
# Pad documents to 1
max_length = 1
padded_docs15 = pad_sequences(encoded_docs15, maxlen=max_length, padding='post')
vocab_size15

# POST  ISSUE TYPE
docs16 = df_train['POST  ISSUE TYPE'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs16)
vocab_size16 = len(t.word_index) + 1
# Label encode the documents
encoded_docs16 = t.texts_to_sequences(docs16)
# Pad documents to 1
max_length = 1
padded_docs16 = pad_sequences(encoded_docs16, maxlen=max_length, padding='post')
vocab_size16

def bootstrap_sample_generator(batch_size):
    while True:
        batch_idx = np.random.choice(
            851, batch_size)
        yield ({'cat_inputs1': padded_docs1[batch_idx],
                'cat_inputs2': padded_docs2[batch_idx],
                'cat_inputs3': padded_docs3[batch_idx],
                'cat_inputs4': padded_docs4[batch_idx],
                'cat_inputs5': padded_docs5[batch_idx],
                'cat_inputs6': padded_docs6[batch_idx],
                'cat_inputs7': padded_docs7[batch_idx],
                'cat_inputs8': padded_docs8[batch_idx],
                'cat_inputs9': padded_docs9[batch_idx],
                'cat_inputs10': padded_docs10[batch_idx],
                'cat_inputs11': padded_docs11[batch_idx],
                'cat_inputs12': padded_docs12[batch_idx],
                'cat_inputs13': padded_docs13[batch_idx],
                'cat_inputs14': padded_docs14[batch_idx],
                'cat_inputs15': padded_docs15[batch_idx],
                'cat_inputs16': padded_docs16[batch_idx],
                'numeric_inputs': issues_count[batch_idx]
                },
               {'output': labels[batch_idx] })

# Embedding size rule from fastai
def emb_sz_rule(n_cat):
    return min(600, round(1.6 * n_cat**0.56))
p = .1

# Start modelling
# Make a different input layer for each category
cat_inputs1 = tf.keras.layers.Input((1,), name='cat_inputs1')
cat_inputs2 = tf.keras.layers.Input((1,), name='cat_inputs2')
cat_inputs3 = tf.keras.layers.Input((1,), name='cat_inputs3')
cat_inputs4 = tf.keras.layers.Input((1,), name='cat_inputs4')
cat_inputs5 = tf.keras.layers.Input((1,), name='cat_inputs5')
cat_inputs6 = tf.keras.layers.Input((1,), name='cat_inputs6')
cat_inputs7 = tf.keras.layers.Input((1,), name='cat_inputs7')
cat_inputs8 = tf.keras.layers.Input((1,), name='cat_inputs8')
cat_inputs9 = tf.keras.layers.Input((1,), name='cat_inputs9')
cat_inputs10 = tf.keras.layers.Input((1,), name='cat_inputs10')
cat_inputs11 = tf.keras.layers.Input((1,), name='cat_inputs11')
cat_inputs12 = tf.keras.layers.Input((1,), name='cat_inputs12')
cat_inputs13 = tf.keras.layers.Input((1,), name='cat_inputs13')
cat_inputs14 = tf.keras.layers.Input((1,), name='cat_inputs14')
cat_inputs15 = tf.keras.layers.Input((1,), name='cat_inputs15')
cat_inputs16 = tf.keras.layers.Input((1,), name='cat_inputs16')
numeric_inputs = tf.keras.layers.Input((1,), name='numeric_inputs')

# Make an embedding layer for each categorical value
# MONTH
embedding_layer = tf.keras.layers.Embedding(
    vocab_size1,
    emb_sz_rule(vocab_size1),
    input_length=1)
cat_x1 = embedding_layer(cat_inputs1)
global_ave1 = tf.keras.layers.GlobalAveragePooling1D()(cat_x1)
global_max1 = tf.keras.layers.GlobalMaxPool1D()(cat_x1)

x1 = tf.keras.layers.Concatenate()([global_ave1, global_max1])

# HOUR
embedding_layer = tf.keras.layers.Embedding(
    vocab_size2,
    emb_sz_rule(vocab_size2),
    input_length=1)
cat_x2 = embedding_layer(cat_inputs2)
global_ave2 = tf.keras.layers.GlobalAveragePooling1D()(cat_x2)
global_max2 = tf.keras.layers.GlobalMaxPool1D()(cat_x2)

x2 = tf.keras.layers.Concatenate()([global_ave2, global_max2])

# CITY - VENUE
embedding_layer = tf.keras.layers.Embedding(
    vocab_size3,
    emb_sz_rule(vocab_size3),
    input_length=1)
cat_x3 = embedding_layer(cat_inputs3)
global_ave3 = tf.keras.layers.GlobalAveragePooling1D()(cat_x3)
global_max3 = tf.keras.layers.GlobalMaxPool1D()(cat_x3)

x3 = tf.keras.layers.Concatenate()([global_ave3, global_max3])

# HOME
embedding_layer = tf.keras.layers.Embedding(
    vocab_size4,
    emb_sz_rule(vocab_size4),
    input_length=1)
cat_x4 = embedding_layer(cat_inputs4)
global_ave4 = tf.keras.layers.GlobalAveragePooling1D()(cat_x4)
global_max4 = tf.keras.layers.GlobalMaxPool1D()(cat_x4)

x4 = tf.keras.layers.Concatenate()([global_ave4, global_max4])

# AWAY
embedding_layer = tf.keras.layers.Embedding(
    vocab_size5,
    emb_sz_rule(vocab_size5),
    input_length=1)
cat_x5 = embedding_layer(cat_inputs5)
global_ave5 = tf.keras.layers.GlobalAveragePooling1D()(cat_x5)
global_max5 = tf.keras.layers.GlobalMaxPool1D()(cat_x5)

x5 = tf.keras.layers.Concatenate()([global_ave5, global_max5])

# COMPETITION'
embedding_layer = tf.keras.layers.Embedding(
    vocab_size6,
    emb_sz_rule(vocab_size6),
    input_length=1)
cat_x6 = embedding_layer(cat_inputs6)
global_ave6 = tf.keras.layers.GlobalAveragePooling1D()(cat_x6)
global_max6 = tf.keras.layers.GlobalMaxPool1D()(cat_x6)

x6 = tf.keras.layers.Concatenate()([global_ave6, global_max6])

# ROUND
embedding_layer = tf.keras.layers.Embedding(
    vocab_size7,
    emb_sz_rule(vocab_size7),
    input_length=1)
cat_x7 = embedding_layer(cat_inputs7)
global_ave7 = tf.keras.layers.GlobalAveragePooling1D()(cat_x7)
global_max7 = tf.keras.layers.GlobalMaxPool1D()(cat_x7)

x7 = tf.keras.layers.Concatenate()([global_ave7, global_max7])

# SYSTEM N°
embedding_layer = tf.keras.layers.Embedding(
    vocab_size8,
    emb_sz_rule(vocab_size8),
    input_length=1)
cat_x8 = embedding_layer(cat_inputs8)
global_ave8 = tf.keras.layers.GlobalAveragePooling1D()(cat_x8)
global_max8 = tf.keras.layers.GlobalMaxPool1D()(cat_x8)

x8 = tf.keras.layers.Concatenate()([global_ave8, global_max8])

# OPERATOR
embedding_layer = tf.keras.layers.Embedding(
    vocab_size9,
    emb_sz_rule(vocab_size9),
    input_length=1)
cat_x9 = embedding_layer(cat_inputs9)
global_ave9 = tf.keras.layers.GlobalAveragePooling1D()(cat_x9)
global_max9 = tf.keras.layers.GlobalMaxPool1D()(cat_x9)

x9 = tf.keras.layers.Concatenate()([global_ave9, global_max9])

# STABILIZATION RESS. NEEDS
embedding_layer = tf.keras.layers.Embedding(
    vocab_size10,
    emb_sz_rule(vocab_size10),
    input_length=1)
cat_x10 = embedding_layer(cat_inputs10)
global_ave10 = tf.keras.layers.GlobalAveragePooling1D()(cat_x10)
global_max10 = tf.keras.layers.GlobalMaxPool1D()(cat_x10)

x10 = tf.keras.layers.Concatenate()([global_ave10, global_max10])


# WEATHER CONDITIONS
embedding_layer = tf.keras.layers.Embedding(
    vocab_size11,
    emb_sz_rule(vocab_size11),
    input_length=1)
cat_x11 = embedding_layer(cat_inputs11)
global_ave11 = tf.keras.layers.GlobalAveragePooling1D()(cat_x11)
global_max11 = tf.keras.layers.GlobalMaxPool1D()(cat_x11)

x11 = tf.keras.layers.Concatenate()([global_ave11, global_max11])

# STADIUM CONDITIONS
embedding_layer = tf.keras.layers.Embedding(
    vocab_size12,
    emb_sz_rule(vocab_size12),
    input_length=1)
cat_x12 = embedding_layer(cat_inputs12)
global_ave12 = tf.keras.layers.GlobalAveragePooling1D()(cat_x12)
global_max12 = tf.keras.layers.GlobalMaxPool1D()(cat_x12)

x12 = tf.keras.layers.Concatenate()([global_ave12, global_max12])

# INTERNET CONDITIONS
embedding_layer = tf.keras.layers.Embedding(
    vocab_size13,
    emb_sz_rule(vocab_size13),
    input_length=1)
cat_x13 = embedding_layer(cat_inputs13)
global_ave13 = tf.keras.layers.GlobalAveragePooling1D()(cat_x13)
global_max13 = tf.keras.layers.GlobalMaxPool1D()(cat_x13)

x13 = tf.keras.layers.Concatenate()([global_ave13, global_max13])

# ISSUE TYPE
embedding_layer = tf.keras.layers.Embedding(
    vocab_size14,
    emb_sz_rule(vocab_size14),
    input_length=1)
cat_x14 = embedding_layer(cat_inputs14)
global_ave14 = tf.keras.layers.GlobalAveragePooling1D()(cat_x14)
global_max14 = tf.keras.layers.GlobalMaxPool1D()(cat_x14)

x14 = tf.keras.layers.Concatenate()([global_ave14, global_max14])

# LIVE TRACKING ISSUE TYPE
embedding_layer = tf.keras.layers.Embedding(
    vocab_size15,
    emb_sz_rule(vocab_size15),
    input_length=1)
cat_x15 = embedding_layer(cat_inputs15)
global_ave15 = tf.keras.layers.GlobalAveragePooling1D()(cat_x15)
global_max15 = tf.keras.layers.GlobalMaxPool1D()(cat_x15)

x15 = tf.keras.layers.Concatenate()([global_ave15, global_max15])

# POST  ISSUE TYPE
embedding_layer = tf.keras.layers.Embedding(
    vocab_size16,
    emb_sz_rule(vocab_size16),
    input_length=1)
cat_x16 = embedding_layer(cat_inputs16)
global_ave16 = tf.keras.layers.GlobalAveragePooling1D()(cat_x16)
global_max16 = tf.keras.layers.GlobalMaxPool1D()(cat_x16)

x16 = tf.keras.layers.Concatenate()([global_ave16, global_max16])

# Concatenating numeric layer and categorical layer together
x = tf.keras.layers.Concatenate()([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                                   x11, x12, x13, x14, x15, x16, numeric_inputs])

# Neural Network
x = tf.keras.layers.Dropout(p)(x)
x = tf.keras.layers.Dense(100, activation='relu')(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(p)(x)
x = tf.keras.layers.Dense(20, activation='relu')(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(p)(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(p)(x)
out = tf.keras.layers.Dense(4, activation='softmax', name='output')(x)

# Compile our model
model = tf.keras.models.Model(inputs=[cat_inputs1, cat_inputs2, cat_inputs3, cat_inputs4, cat_inputs5, cat_inputs6, cat_inputs7,
                                      cat_inputs8, cat_inputs9, cat_inputs10, cat_inputs11, cat_inputs12, cat_inputs13, cat_inputs14,
                                      cat_inputs15, cat_inputs16, numeric_inputs], outputs=out)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
########################################################################################################################
# Train
# Fit the model
# batch_size = 16
# model.fit_generator(
#     bootstrap_sample_generator(batch_size),
#     steps_per_epoch=1000// batch_size,
#     epochs=5,
#     max_queue_size=10,)
batch_size = 32
model.fit_generator(
    bootstrap_sample_generator(batch_size),
    steps_per_epoch=1000// batch_size,
    epochs=50,
    max_queue_size=10,)

# Plotting our model
plot_model(model, show_shapes=True, show_layer_names=True)

# Example
def make_prediction100(rank):
    X_test = []
    L = []
    R = []
    predicted_grade = []
    for i in range(100):
        X_test.append({'cat_inputs1': padded_docs1[rank+i],
                    'cat_inputs2': padded_docs2[rank+i],
                    'cat_inputs3': padded_docs3[rank+i],
                    'cat_inputs4': padded_docs4[rank+i],
                    'cat_inputs5': padded_docs5[rank+i],
                    'cat_inputs6': padded_docs6[rank+i],
                    'cat_inputs7': padded_docs7[rank+i],
                    'cat_inputs8': padded_docs8[rank+i],
                    'cat_inputs9': padded_docs9[rank+i],
                    'cat_inputs10': padded_docs10[rank+i],
                    'cat_inputs11': padded_docs11[rank+i],
                    'cat_inputs12': padded_docs12[rank+i],
                    'cat_inputs13': padded_docs13[rank+i],
                    'cat_inputs14': padded_docs14[rank+i],
                    'cat_inputs15': padded_docs15[rank+i],
                    'cat_inputs16': padded_docs16[rank+i],
                    'numeric_inputs': issues_count[rank+i]
                    })
    print(X_test)
    for i  in range(len(X_test)):
        print(model.predict(X_test[i]))
        L.append(model.predict(X_test[i]))
    for i in range(len(X_test)):
        print('Rounded prediction',np.argmax(L[i]))
        R.append(np.argmax(L[i]))
    for i in range(len(X_test)):
        if R[i]==0:
            print('A')
            predicted_grade.append('A')
        elif R[i]==1:
            print('B')
            predicted_grade.append('B')
        elif R[i]==2:
            print('C')
            predicted_grade.append('C')
        else:
            print('D')
            predicted_grade.append('D')
########################################################################################################################
# Test
########################################################################################################################
# To make predictions
# Read csv file
sim = pd.read_csv('/Users/mengwenwu/PycharmProjects/Stage22/L1_simu.csv')
sim.head()
sim.columns

# Some formatting
# Convert integers to strings
sim['MONTH'] = sim['MONTH'].astype('string')
# Standardize Issues Count
sim['ISSUES COUNT'] = stats.zscore(sim['ISSUES COUNT'])
sim['ROUND'] = sim['ROUND'].astype('str')
# Numerical feature
issues_count = sim['ISSUES COUNT']
issues_count = sim[['ISSUES COUNT']].values
issues_count

# MONTH
docs1 = sim['MONTH'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs1)
vocab_size1 = len(t.word_index) + 1
# Label encode the documents
encoded_docs1 = t.texts_to_sequences(docs1)
# Pad documents to 1
max_length = 1
padded_docs1 = pad_sequences(encoded_docs1, maxlen=max_length, padding='post')
vocab_size1

# HOUR
docs2 = sim['HOUR'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs2)
vocab_size2 = len(t.word_index) + 1
# Label encode the documents
encoded_docs2 = t.texts_to_sequences(docs2)
# Pad documents to 1
max_length = 1
padded_docs2 = pad_sequences(encoded_docs2, maxlen=max_length, padding='post')
vocab_size2

# CITY - VENUE
docs3 = sim['CITY - VENUE'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs3)
vocab_size3 = len(t.word_index) + 1
# Label encode the documents
encoded_docs3 = t.texts_to_sequences(docs3)
# Pad documents to 1
max_length = 1
padded_docs3 = pad_sequences(encoded_docs3, maxlen=max_length, padding='post')
vocab_size3

# HOME
docs4 = sim['HOME'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs4)
vocab_size4 = len(t.word_index) + 1
# Label encode the documents
encoded_docs4 = t.texts_to_sequences(docs4)
# Pad documents to 1
max_length = 1
padded_docs4 = pad_sequences(encoded_docs4, maxlen=max_length, padding='post')
vocab_size4

# AWAY
docs5 = sim['AWAY'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs5)
vocab_size5 = len(t.word_index) + 1
# Label encode the documents
encoded_docs5 = t.texts_to_sequences(docs5)
# Pad documents to 1
max_length = 1
padded_docs5 = pad_sequences(encoded_docs5, maxlen=max_length, padding='post')
vocab_size5

# COMPETITION
docs6 = sim['COMPETITION'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs6)
vocab_size6 = len(t.word_index) + 1
# Label encode the documents
encoded_docs6 = t.texts_to_sequences(docs6)
# Pad documents to 1
max_length = 1
padded_docs6 = pad_sequences(encoded_docs6, maxlen=max_length, padding='post')
vocab_size6

# ROUND
docs7 = sim['ROUND'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs7)
vocab_size7 = len(t.word_index) + 1
# Label encode the documents
encoded_docs7 = t.texts_to_sequences(docs7)
# Pad documents to 1
max_length = 1
padded_docs7 = pad_sequences(encoded_docs7, maxlen=max_length, padding='post')
vocab_size7

# SYSTEM N°
docs8 = sim['SYSTEM N°'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs8)
vocab_size8 = len(t.word_index) + 1
# Label encode the documents
encoded_docs8 = t.texts_to_sequences(docs8)
# Pad documents to 1
max_length = 1
padded_docs8 = pad_sequences(encoded_docs8, maxlen=max_length, padding='post')
vocab_size8

# OPERATOR
docs9 = sim['OPERATOR'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs9)
vocab_size9 = len(t.word_index) + 1
# Label encode the documents
encoded_docs9 = t.texts_to_sequences(docs9)
# Pad documents to 1
max_length = 1
padded_docs9 = pad_sequences(encoded_docs9, maxlen=max_length, padding='post')
vocab_size9

# STABILIZATION RESS. NEEDS
docs10 = sim['STABILIZATION RESS. NEEDS'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs10)
vocab_size10 = len(t.word_index) + 1
# Label encode the documents
encoded_docs10 = t.texts_to_sequences(docs10)
# Pad documents to 1
max_length = 1
padded_docs10 = pad_sequences(encoded_docs10, maxlen=max_length, padding='post')
vocab_size10

# WEATHER CONDITIONS
docs11 = sim['WEATHER CONDITIONS'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs11)
vocab_size11 = len(t.word_index) + 1
# Label encode the documents
encoded_docs11 = t.texts_to_sequences(docs11)
# Pad documents to 1
max_length = 1
padded_docs11 = pad_sequences(encoded_docs11, maxlen=max_length, padding='post')
vocab_size11

# STADIUM CONDITIONS
docs12 = sim['STADIUM CONDITIONS'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs12)
vocab_size12 = len(t.word_index) + 1
# Label encode the documents
encoded_docs12 = t.texts_to_sequences(docs12)
# Pad documents to 1
max_length = 1
padded_docs12 = pad_sequences(encoded_docs12, maxlen=max_length, padding='post')
vocab_size12

# INTERNET CONDITIONS
sim['INTERNET CONDITIONS']=sim['INTERNET CONDITIONS'].astype('str')
docs13 = sim['INTERNET CONDITIONS'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs13)
vocab_size13 = len(t.word_index) + 1
# Label encode the documents
encoded_docs13 = t.texts_to_sequences(docs13)
# Pad documents to 1
max_length = 1
padded_docs13 = pad_sequences(encoded_docs13, maxlen=max_length, padding='post')
vocab_size13

# ISSUE TYPE
docs14 = sim['ISSUE TYPE'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs14)
vocab_size14 = len(t.word_index) + 1
# Label encode the documents
encoded_docs14 = t.texts_to_sequences(docs14)
# Pad documents to 1
max_length = 1
padded_docs14 = pad_sequences(encoded_docs14, maxlen=max_length, padding='post')
vocab_size14

# LIVE TRACKING ISSUE TYPE
docs15 = sim['LIVE TRACKING ISSUE TYPE'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs15)
vocab_size15 = len(t.word_index) + 1
# Label encode the documents
encoded_docs15 = t.texts_to_sequences(docs15)
# Pad documents to 1
max_length = 1
padded_docs15 = pad_sequences(encoded_docs15, maxlen=max_length, padding='post')
vocab_size15

# POST  ISSUE TYPE
docs16 = sim['POST  ISSUE TYPE'].values
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs16)
vocab_size16 = len(t.word_index) + 1
# Label encode the documents
encoded_docs16 = t.texts_to_sequences(docs16)
# Pad documents to 1
max_length = 1
padded_docs16 = pad_sequences(encoded_docs16, maxlen=max_length, padding='post')
vocab_size16

def make_predictions(rank):
    X_test = []
    L = []
    R = []
    predicted_grade = []
    for i in range(len(sim)):
        X_test.append({'cat_inputs1': padded_docs1[rank+i],
                    'cat_inputs2': padded_docs2[rank+i],
                    'cat_inputs3': padded_docs3[rank+i],
                    'cat_inputs4': padded_docs4[rank+i],
                    'cat_inputs5': padded_docs5[rank+i],
                    'cat_inputs6': padded_docs6[rank+i],
                    'cat_inputs7': padded_docs7[rank+i],
                    'cat_inputs8': padded_docs8[rank+i],
                    'cat_inputs9': padded_docs9[rank+i],
                    'cat_inputs10': padded_docs10[rank+i],
                    'cat_inputs11': padded_docs11[rank+i],
                    'cat_inputs12': padded_docs12[rank+i],
                    'cat_inputs13': padded_docs13[rank+i],
                    'cat_inputs14': padded_docs14[rank+i],
                    'cat_inputs15': padded_docs15[rank+i],
                    'cat_inputs16': padded_docs16[rank+i],
                    'numeric_inputs': issues_count[rank+i]
                    })
    print(X_test)
    for i  in range(len(X_test)):
        print(model.predict(X_test[i]))
        L.append(model.predict(X_test[i]))
    for i in range(len(X_test)):
        print('Rounded prediction',np.argmax(L[i]))
        R.append(np.argmax(L[i]))
    for i in range(len(X_test)):
        if R[i]==0:
            print('A')
            predicted_grade.append('A')
        elif R[i]==1:
            print('B')
            predicted_grade.append('B')
        elif R[i]==2:
            print('C')
            predicted_grade.append('C')
        else:
            print('D')
            predicted_grade.append('D')
    return(predicted_grade)

predicted_grades = make_predictions(0)
sim['PREDICTED_GRADE'] = pd.Series(predicted_grades)
sim.head()
sim.tail()




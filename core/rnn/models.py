from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input, BatchNormalization, concatenate, Lambda
from keras.models import Model, load_model


def Minus(inputs):
    x, y = inputs
    return (x - y)


def BiLSTM(vocab_size, max_sequence_length, max_key_sequence_length, embedding_size=128, hidden_size=32, dropout=0.2,
           multiple=16):
    embedding_layer_1 = Embedding(vocab_size + 1, embedding_size, input_length=8)
    embedding_layer_2 = Embedding(vocab_size + 1, embedding_size, input_length=max_sequence_length)
    lstm_layer1 = LSTM(hidden_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)
    lstm_layer2 = LSTM(hidden_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=False)

    lstm_layer3 = LSTM(hidden_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)
    lstm_layer4 = LSTM(hidden_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=False)

    seq_input_1 = Input(shape=(max_key_sequence_length,), dtype='int32')
    y1 = lstm_layer1(embedding_layer_1(seq_input_1))
    y1 = lstm_layer2(y1)

    seq_input_2 = Input(shape=(max_key_sequence_length,), dtype='int32')
    y2 = lstm_layer1(embedding_layer_1(seq_input_2))
    y2 = lstm_layer2(y2)

    seq_input_3 = Input(shape=(max_sequence_length,), dtype='int32')
    y3 = lstm_layer3(embedding_layer_2(seq_input_3))
    y3 = lstm_layer4(y3)

    seq_input_4 = Input(shape=(max_sequence_length,), dtype='int32')
    y4 = lstm_layer3(embedding_layer_2(seq_input_4))
    y4 = lstm_layer4(y4)

    rate = Input(shape=(2 * multiple,), dtype='float32')
    reverse = Input(shape=(multiple,), dtype='float32')

    minus = Lambda(Minus)

    y = BatchNormalization()(concatenate(
        [y1, y2, y3, y4, minus([y3, y4]), minus([y1, y2]), rate, reverse]))
    y = Dense(64)(y)
    y = Dense(16)(y)
    preds = Dense(1, activation='sigmoid')(y)

    model = Model(inputs=[seq_input_1, seq_input_2, rate, seq_input_3, seq_input_4, reverse], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=["accuracy"])
    model.summary()
    return model


"""
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, max_sequence_length, batch_size=32, embedding_size=128, hidden_size=32, dropout=0.2,
                 multiple=16):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.msl = max_sequence_length
        self.multiple = multiple

        # embedding layer
        self.embed_seq = nn.Embedding(self.vocab_size + 1, embedding_size)
        self.embed_keyword_seq = nn.Embedding(self.vocab_size + 1, embedding_size)
        print(type(self.embed_seq.weight))

        # LSTM layer
        self.lstm_seq1 = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, dropout=dropout)
        self.lstm_seq2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)

        self.lstm_key_seq1 = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, dropout=dropout)
        self.lstm_key_seq2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)

        # dense layer & normalization
        # self.normalization = nn.BatchNorm1d()
        self.output = nn.Sequential(
            nn.Linear(6 * hidden_size + 3 * multiple, 64), #
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.Linear(16, 2),
            nn.Softmax()
        )

    def forward(self, mag, aminer, jaccard, keyword_mag, keyword_aminer, inverse):
        mag = self.embed_seq(mag)
        aminer = self.embed_seq(aminer)
        keyword_mag = self.embed_keyword_seq(keyword_mag)
        keyword_aminer = self.embed_keyword_seq(keyword_aminer)
        jaccard = jaccard.repeat(1, 2 * self.multiple)
        inverse = inverse.repeat(1, self.multiple)
        mag, _ = self.lstm_seq1(mag)
        mag, _ = self.lstm_seq2(mag)
        aminer, _ = self.lstm_seq1(aminer)
        aminer, _ = self.lstm_seq2(aminer)
        keyword_mag, _ = self.lstm_key_seq1(keyword_mag)
        keyword_mag, _ = self.lstm_key_seq2(keyword_mag)
        keyword_aminer, _ = self.lstm_key_seq1(keyword_aminer)
        keyword_aminer, _ = self.lstm_key_seq2(keyword_aminer)
        minus = keyword_mag[:, -1, :] - keyword_aminer[:, -1, :]
        minus_key = mag[:, -1, :] - aminer[:, -1, :]
        concat_input = torch.cat(
            (minus,
             minus_key,
             jaccard,
             inverse,
             mag[:, -1, :],
             aminer[:, -1, :],
             keyword_mag[:, -1, :],
             keyword_aminer[:, -1, :],
             ), dim=1)

        output = self.output(concat_input)
        # output = self.sigmoid(output)
        return output
"""

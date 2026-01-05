# The encoder RNN reads stroke input sequences and extracts a context vector
class Encoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        # each stroke point has 128 features
        # dimensionality of hidden state
        # Batch, Seq, Feature
        self.rnn =nn.LSTM(input_size=128,hidden_size=d,batch_first=True)

    def forward(self, x):
        # replace PAD_VALUEs with 0 before feeding to LSTM to prevent numerical issues
        x_processed = torch.where(x ==PAD_VALUE, torch.zeros_like(x), x)

        # LSTM returns output_sequence,hidden_state,cell_state
        return self.rnn(x_processed)  
    
    class Decoder(nn.Module):
        # takes a sequence of token ids y and a previous lstm hidden state
        # looks up embeddings for each token then feeds them through an lstm
        # produces a hidden vector at every timestep and projects it to vocab logits
        # returns the logits for all timesteps plus the updated hidden state for continued decoding
        def __init__(self, vocab=INFIX_VOCAB_SIZE, d=128):
            super().__init__()
            self.emb = nn.Embedding(vocab, d, padding_idx=PAD_IDX)
            self.rnn =nn.LSTM(input_size=d,hidden_size=d, batch_first=True)
            self.fc = nn.Linear(d, vocab)

        def forward(self, y, hidden):
            e = self.emb(y)
            out, hid = self.rnn(e,hidden)
            out = self.fc(out)
            return out, hid
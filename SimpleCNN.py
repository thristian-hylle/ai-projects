class SimpleCNN(nn.Module):

    #Baseline CNN model trained from scratch Operates on 16 frames outputs per frame featuresthen averages them to produce a video level prediction

    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()

        # A simple 3-layer CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32,kernel_size=3,padding=1), # 3 to 32
            nn.ReLU(), # not lineariltty
            nn.MaxPool2d(2), # 224 to 112

            nn.Conv2d(32, 64,kernel_size=3,padding=1), #32 to 64
            nn.ReLU(),
            nn.MaxPool2d(2),# 112 to 56

            nn.Conv2d(64,128,kernel_size=3,padding=1),# 64 to 128
            nn.ReLU(),
            nn.MaxPool2d(2),#56 to 28
        )

        # FC classifier (video-level)
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),#flatten conv map to hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),# regularization
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B,T,C,H,W =x.shape #(batch,time,channels,height,width)
        x = x.view(B * T,C,H,W)  # merge batch+time for CNN

        # Extract per-frame features
        feats = self.features(x)  # (B*T, 128, 28, 28)
        feats =feats.reshape(B, T,-1)  # reshape back to (B,T,features_dim)

        # temporal pooling averrage frame features into one vector per video
        video_feats= feats.mean(dim=1)  #(B,features_dim)

        # classify entire video
        out = self.fc(video_feats)
        return out
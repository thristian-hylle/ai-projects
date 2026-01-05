
class ResNet18(nn.Module):

    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet18, self).__init__()

        # load standard torchvision resnet18
        self.resnet = resnet18(pretrained=pretrained)

        # Remove the final fully-connected layer, keep feature extractor
        self.resnet =nn.Sequential(*list(self.resnet.children())[:-1])
        # Output shape becomes (B * T, 512, 1, 1)

        # new classifier for video level prediction
        self.classifier = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # merge batch and time (B*T, 3,224, 224)
        x = x.view(B * T, C, H, W)

        # Extract frame level features
        feats =self.resnet(x)  # (B*T, 512,1, 1)
        feats = feats.view(B,T, 512)  #(B,T,512)

        # average across frames video representation
        video_feats= feats.mean(dim=1)  # (B, 512)

        # classify the whole video
        out = self.classifier(video_feats) #(B,num_classes)
        return out
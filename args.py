import argparse

parser = argparse.ArgumentParser()

### generic args ###
parser.add_argument("--device", type=str,
                    default="cuda:0")  # device to train on
parser.add_argument("--workers", type=int, default=4)  # number of workers
parser.add_argument("--wandb", action="store_true")  # use wandb
parser.add_argument("--jobname", type=str)  # device to train on


### data args ###
parser.add_argument("--feature", type=str, default="melspec")  # feature to use
parser.add_argument(
    "--traindir",
    type=str,
    default="/users/local/i21moumm/dcase23/Development_Set/Training_Set",
)  # root dir for the training dataset
parser.add_argument(
    "--valdir",
    type=str,
    default="/users/local/i21moumm/dcase23/Development_Set/Validation_Set",
)  # root dir for the validation dataset
parser.add_argument(
    "--h5file", type=str, default="train.h5"
)  # name of train H5 file to use (for parallel computing)


### training args ###
# model to use (at the moment only resnet is supported)
parser.add_argument("--model", type=str, default='resnet')
parser.add_argument(
    "--method", type=str, default="scl"
)  # whether to use labels or not for training representations ['scl', 'ssl]
parser.add_argument(
    "--bs", type=int, default=128
)  # batch size for representation learning
parser.add_argument("--wd", type=float, default=1e-4)  # weight decay
parser.add_argument("--momentum", type=float, default=0.9)  # sgd momentum
parser.add_argument("--lr", type=float, default=1e-2)  # learning rate
parser.add_argument(
    "--epochs", type=int, default=50
)  # nb of epochs to train the feature extractor on the training set


### loss function args ###
# temperature for cosine sim
parser.add_argument("--tau", type=float, default=0.06)
parser.add_argument(
    "--margin", type=float, default=0.4
)  # margin parameter for Angular Margin Loss component
parser.add_argument(
    "--alpha", type=float, default=0.5
)  # alpha weight parameter for ACL


### finetuning args ###
parser.add_argument(
    "--ft", type=int, default=0
    # number of layers to finetune; 1, 2 or 3 layers (for the ResNet we are using)
)
parser.add_argument(
    "--ftlr", type=float, default=1e-2
)  # learning rate for finetuning on support set
parser.add_argument(
    "--ftepochs", type=int, default=20
)  # nb of epochs to finetune on support set
parser.add_argument(
    "--ftbs", type=int, default=32
)  # batch size for fine tuning prototypes
# batch size for query prediction
parser.add_argument("--qbs", type=int, default=16)
parser.add_argument("--adam", action="store_true")  # use adam instead of sgd
# scheduler step size for adam
parser.add_argument("--step", type=int, default=10)
# evaluation scheduler gamma
parser.add_argument("--gamma", type=float, default=0.5)


### few shot args ###
parser.add_argument("--nshot", type=int, default=5)  # number of shots


### audio ###
parser.add_argument("--sr", type=int, default=22050)  # sampling rate for audio
parser.add_argument(
    "--len", type=int, default=200
)  # segment duration for training in ms


### mel spec parameters ###
parser.add_argument("--nmels", type=int, default=128)  # number of mels
parser.add_argument("--nfft", type=int, default=512)  # size of FFT
# hop between STFT windows
parser.add_argument("--hoplen", type=int, default=128)
parser.add_argument("--fmax", type=int, default=11025)  # fmax
parser.add_argument("--fmin", type=int, default=50)  # fmin


### data augmentation ###
parser.add_argument(
    "--tratio", type=float, default=0.6
)  # time ratio for spectrogram crop
parser.add_argument(
    "--noise", type=float, default=0.01
)  # standard deviation for additive white gaussian noise
parser.add_argument(
    "--comp", type=float, default=0.75
)  # compander coefficient to compress signal
parser.add_argument(
    "--fshift", type=int, default=10
)  # frequency bands to shift upwards


args = parser.parse_args()

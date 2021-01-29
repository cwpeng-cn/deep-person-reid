import sys
sys.path.append("../")
import torch
import torchreid
from torchreid.utils.torchtools import load_pretrained_weights

torch.manual_seed(0)

# Each batch contains batch_size*seq_len images
datamanager = torchreid.data.VideoDataManager(
    root='../../../datasets',
    sources='prid2011',
    sample_method="random",
    height=384,
    width=128,
    transforms=['random_flip', 'random_crop'],
    combineall=False,
    batch_size_train=12,  # number of tracklets
    seq_len=8  # number of images in each tracklet
)

model = torchreid.models.build_model(
    name='resnet_video',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    seq_num=8,
    d_ff=2048,
    h=1,
    droprate=0.1,
    N=2
)

# load_pretrained_weights(model=model, weight_path="log/resnet50-softmax-prid2011/model/model.pth.tar-43")
model = model.cuda()
optimizer = torchreid.optim.build_optimizer(
    model, optim='adam', lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer, lr_scheduler='single_step', stepsize=20)

engine = torchreid.engine.VideoSoftmaxATTEngine(
    datamanager, model, optimizer, scheduler=scheduler
)

engine.run(
    max_epoch=100,
    save_dir='log/resnet50-video-softmax-prid2011',
    print_freq=10, start_eval=10, eval_freq=3
)

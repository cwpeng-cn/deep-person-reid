import torch
import torchreid
from torchreid.utils.torchtools import load_pretrained_weights

torch.manual_seed(5)

# Each batch contains batch_size*seq_len images
datamanager = torchreid.data.VideoDataManager(
    root='../',
    sources='prid2011_gait2_cropped_newsplit',
    sample_method="random",
    transforms=None,
    height=256,
    width=128,
    combineall=False,
    batch_size_train=16,  # number of tracklets
    seq_len=15  # number of images in each tracklet
)


model = torchreid.models.build_model(
    name='resnet_video_fcatt',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    seq_num=15,
    d_ff=2048,
    h=1,
    droprate=0.1,
    N=2
)

# load_pretrained_weights(model=model, weight_path="log/resnet50-softmax-prid2011/model/model.pth.tar-43")
model = model.cuda()
optimizer = torchreid.optim.build_optimizer(
    model, optim='adam', lr=0.00003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer, lr_scheduler='single_step', stepsize=60)

engine = torchreid.engine.VideoSoftmaxATTEngine(
    datamanager, model, optimizer, scheduler=scheduler
)

engine.run(
    max_epoch=150,
    save_dir='log/resnet50-video-softmax-prid2011',
    print_freq=10, start_eval=80, eval_freq=3
)

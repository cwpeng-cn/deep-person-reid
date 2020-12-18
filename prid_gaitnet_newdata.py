import torch
import torchreid

torch.manual_seed(5)

# Each batch contains batch_size*seq_len images
datamanager = torchreid.data.VideoDataManager(
    root='../',
    sources='prid2011_gait2_cropped',
    sample_method="random",
    transforms=None,
    height=256,
    width=128,
    combineall=False,
    batch_size_train=16,  # number of tracklets
    seq_len=15  # number of images in each tracklet
)

model = torchreid.models.build_model(
    name='gaitnet',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    seq_num=15,
)

model = model.cuda()
optimizer = torchreid.optim.build_optimizer(
    model, optim='adam', lr=0.0001
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer, lr_scheduler='single_step', stepsize=40)

engine = torchreid.engine.VideoSoftmaxATTEngine(
    datamanager, model, optimizer, scheduler=scheduler
)

engine.run(
    max_epoch=500,
    save_dir='log/resnet50-video-softmax-prid2011',
    print_freq=10, start_eval=200, eval_freq=3
)

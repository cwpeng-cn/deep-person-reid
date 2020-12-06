import torch
import sys
sys.path.append("..")
import torchreid

# Each batch contains batch_size*seq_len images
datamanager = torchreid.data.VideoDataManager(
    root='../../../datasets',
    sources='mars',
    height=256,
    width=128,
    combineall=False,
    batch_size_train=8,  # number of tracklets
    seq_len=15  # number of images in each tracklet
)

model = torchreid.models.build_model(
    name='resnet50_fc512',
    num_classes=datamanager.num_train_pids,
    loss='softmax'
)

model = model.cuda()
optimizer = torchreid.optim.build_optimizer(
    model, optim='adam', lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer, lr_scheduler='single_step', stepsize=20)

engine = torchreid.engine.VideoSoftmaxEngine(
    datamanager, model, optimizer, scheduler=scheduler,
    pooling_method='avg'
)

engine.run(
    max_epoch=60,
    save_dir='log/resnet50-softmax-mars',
    print_freq=10, start_eval=30, eval_freq=2
)

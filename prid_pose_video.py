import torch
import torchreid

torch.manual_seed(5)

# Each batch contains batch_size*seq_len images
datamanager = torchreid.data.VideoDataManager(
    root='../',
    sources='prid2011pose',
    sample_method="random",
    transforms=None,
    height=256,
    width=128,
    combineall=False,
    batch_size_train=8,  # number of tracklets
    seq_len=12  # number of images in each tracklet
)

model = torchreid.models.build_model(
    name='resnet_video',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    seq_num=12,
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
    optimizer, lr_scheduler='single_step', stepsize=40)

engine = torchreid.engine.VideoSoftmaxATTEngine(
    datamanager, model, optimizer, scheduler=scheduler
)

engine.run(
    max_epoch=120,
    save_dir='log/resnet50-video-pose-softmax-prid2011',
    print_freq=50, start_eval=10, eval_freq=3
)

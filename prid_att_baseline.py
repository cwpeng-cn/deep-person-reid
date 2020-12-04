import torch
import torchreid
from torchreid.utils.torchtools import load_pretrained_weights

# Each batch contains batch_size*seq_len images
datamanager = torchreid.data.VideoDataManager(
    root='../../datasets',
    sources='prid2011',
    height=256,
    width=128,
    combineall=False,
    batch_size_train=10,  # number of tracklets
    seq_len=15  # number of images in each tracklet
)

model = torchreid.models.build_model(
    name='resnet50_fc512_att',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    seq_num=15,
    d_ff=2048,
    h=1,
    droprate=0.1,
    N=2
)

# load_pretrained_weights(model=model, weight_path="log/resnet50-softmax-prid2011/model/model.pth.tar-43")
model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model, optim='adam', lr=1e-4, staged_lr=True, new_layers=['fc', 'classifier', "encoder", "attn", "ff"],
    base_lr_mult=0.1
)

print(model)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer, lr_scheduler='single_step', stepsize=20)

engine = torchreid.engine.VideoSoftmaxATTEngine(
    datamanager, model, optimizer, scheduler=scheduler
)

engine.run(
    max_epoch=100,
    save_dir='log/resnet50att-softmax-prid2011',
    print_freq=10, start_eval=40, eval_freq=2
)

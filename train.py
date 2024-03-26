import torch
import open_clip
from oc_clip import OCClip
from dataset import Image_Dataset, CLEVER
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
from helper import UEMA

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}!\n")

parser = argparse.ArgumentParser(
    prog="CLIPLIKE",
    description="Train model to output CLIP-vector from slots"
)
parser.add_argument("-p", "--dataset_path", type=str, help="path to the datasets image files",
                    default="../datasets/bedroom")
parser.add_argument("-t", "--filetype", type=str, help="image filetype e.g. jpg png ...",
                    default="webp")
parser.add_argument("-b", "--batch_size", type=int, help="batch size",
                    default=16)
parser.add_argument("-v", "--verbose", help="print arguments",
                    default=False, action="store_true")
parser.add_argument("-l", "--lr", "--learning_rate", type=float,
                    help="Learning rate", default=1e-4)
parser.add_argument("-m", "--model", type=str,
                    help="Name of open_clip model to use as training traget", default="ViT-H-14")
parser.add_argument("-n", "--num_tf_layers", type=int,
                    help="Number of transformer layers", default=4)
parser.add_argument("-f", "--feature_extractor", type=str,
                    help="feature extractor in the object_centric clip model", default="resnet34")
parser.add_argument("--device", type=str, help="device to use ('cuda', 'cpu')",
                    default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--comment", type=str,
                    help="comment for tensorboard logs", default="")
parser.add_argument("--checkpoint_interval", type=float,
                    help="Interval at which to save a snapshpt (in minutes)", default=10)
parser.add_argument("--total_kimg", type=int,
                    help="Total number of images to train on (in thousands)", default=10_000)

opts = parser.parse_args()
if opts.verbose:
    print(opts)

batch_size = opts.batch_size
total_kimg = opts.total_kimg
loss_uema = UEMA(0.95)
ckpt_path = ""


print("Loading dataset")
if "clever" in opts.dataset_path.lower() or "clevr" in opts.dataset_path.lower():
    dataset = CLEVER(split="train", resolution=(
        224, 224), root_dir=opts.dataset_path)
else:
    dataset = Image_Dataset(split="train", root_dir=opts.dataset_path,
                            resolution=(224, 224), filetype=opts.filetype)

# monolitic_model = timm.create_model("resnet18", pretrained=True, num_classes=0)

clip_name = opts.model
datasets = dict(open_clip.list_pretrained())
monolitic_model, _, _ = open_clip.create_model_and_transforms(
    clip_name, pretrained=datasets[clip_name])
monolitic_model = monolitic_model.to(device)
monolitic_model.requires_grad_(False)

clipdim = monolitic_model.encode_image(
    torch.empty(1, 3, 224, 224).to(device)).shape[-1]

occlip = OCClip(
    device,
    clip_dim=clipdim,
    num_transformer_layers=opts.num_tf_layers,
    oc_type="SA" if isinstance(dataset, CLEVER) else "DINOSAUR",
    feature_extractor=opts.feature_extractor
).to(device)
if ckpt_path:
    ckpt = torch.load(ckpt_path)
    occlip.load_state_dict(ckpt["model_state_dict"])
    try:
        monolitic_model.load_state_dict(ckpt["monolithic_state_dict"])
    except:
        print("Failed loading state dict for monolitic model")
    global_step = ckpt["global_step"]
else:
    global_step = 0


job_suffix = f"occlip_{opts.model}_{opts.feature_extractor}_{opts.num_tf_layers}_{opts.comment}"
writer = SummaryWriter(
    comment=job_suffix)

print(f"Number of training images: {len(dataset)}")
print("Preparing dataloader")
data_loader = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=4)
print("Data done! \n")
optimizer = torch.optim.Adam(
    lr=opts.lr,
    params=list(occlip.parameters())
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=80_000//opts.batch_size, min_lr=1e-7)
loss_fn = torch.nn.MSELoss()
scaler = torch.cuda.amp.GradScaler()

done = False

last_checkpoint = time.time()


def snapshot():
    torch.save({
        "model_state_dict": occlip.state_dict(),
        "global_step": global_step,
        "loss": loss_uema.get(),
        "optimizer_state_dict": optimizer.state_dict()
    }, f"occlip{job_suffix}.ckpt")


writer.add_images(
    "image/real", torch.stack([dataset[i]["image"] for i in range(8)]), global_step)

try:
    while True:
        for batch in data_loader:
            img = batch["image"].to(device)

            global_step += batch_size
            if global_step >= total_kimg * 1000:
                done = True
                break

            if optimizer.param_groups[0]["lr"] < 2e-7:
                print(f"Done due to lr = {optimizer.param_groups[0]['lr']}")
                done = True
                break

            # calculate features
            with torch.autocast(device_type=device):
                optimizer.zero_grad()
                clip_features = monolitic_model.encode_image(
                    img)
                cliplike_features, t = occlip(img)

                loss = loss_fn(clip_features, cliplike_features)

                scaler.scale(loss).backward()
                scaler.step(optimizer=optimizer)
                scaler.update()

                loss_uema.update(loss)

                scheduler.step(loss_uema.get())

                if global_step % 512 == 0:
                    writer.add_scalar("loss", loss_uema.get(), global_step)
                    writer.add_scalar(
                        "debug/scale", scaler.get_scale(), global_step)
                    writer.add_scalar(
                        "hyperparameter/lr", optimizer.param_groups[0]['lr'], global_step)

                if global_step % 512 == 0 or opts.verbose:
                    print(
                        f"{global_step}-images done: loss={(loss_uema.get()*1e4):2.4f}*10^-4, num_bad_epochs={scheduler.num_bad_epochs} / {scheduler.patience }", flush=True)

                if time.time() - last_checkpoint >= 60*opts.checkpoint_interval:
                    snapshot()
                    print("Snapshot saved", flush=True)
                    last_checkpoint = time.time()

        if done:
            break
finally:
    snapshot()

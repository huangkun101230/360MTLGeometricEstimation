import argparse

parser = argparse.ArgumentParser(description="360 MTL Geometric Estimation")

# dataset
parser.add_argument("--dataset", default="3d60", choices=["3d60", "structured3d"],
                    type=str, help="dataset to train on.")

# system settings
parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
parser.add_argument("--using_gpu", type=int, default=0, help="using which gpu")

# model settings
parser.add_argument("--saving_folder", type=str, default="train", help="folder to save the model in")
parser.add_argument("--height", type=int, default=256, help="input image height")
parser.add_argument("--width", type=int, default=512, help="input image width")

# optimization settings
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer, options={"adam, sgd"}')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=12, metavar='LR',
                    help='learning rate decay epoch')
parser.add_argument('--es', type=float, default=12, metavar='N',
                    help='early stopping')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before saving model weights')
parser.add_argument("--amp", default=True, type=bool,
                    help="Use torch.cuda.amp for mixed precision training")
parser.add_argument("--enable_wandb", default=False, type=bool,
                    help="Use wandb for monitering training")


args = parser.parse_args()


def main():
    from trainer import trainer_3d60
    trainer_3d60(args)


if __name__ == "__main__":
    main()

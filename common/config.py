import argparse

argparser = argparse.ArgumentParser(description='Ego4d Social Benchmark')

argparser.add_argument('--source_path', type=str, default='../social-interactions/data/video_imgs', help='Video image directory')
argparser.add_argument('--json_path', type=str, default='../social-interactions/data/json_original', help='Face tracklets directory')
argparser.add_argument('--test_path', type=str, default='../videos_challenge', help='Test set')
argparser.add_argument('--gt_path', type=str, default='../social-interactions/data/result_LAM', help='Groundtruth directory')
argparser.add_argument('--train_file', type=str, default='../social-interactions/data/split/train.list', help='Train list')
argparser.add_argument('--val_file', type=str, default='../social-interactions/data/split/val.list', help='Validation list')
argparser.add_argument('--prior_path',type=str, default='../social-interactions', help='Prior Jsons')
argparser.add_argument('--face_path',type=str, default='../social-interactions/data/face_imgs', help='Prior Jsons')
argparser.add_argument('--mesh', action='store_true', help='Load all prior data')
argparser.add_argument('--prior_head', action='store_true', help='Use head prior')
argparser.add_argument('--prior_landmark', action='store_true', help='Use landmark prior')



argparser.add_argument('--train_stride', type=int, default=3, help='Train subsampling rate')
argparser.add_argument('--val_stride', type=int, default=13, help='Validation subsampling rate')
argparser.add_argument('--test_stride', type=int, default=1, help='Test subsampling rate')
argparser.add_argument('--split',type=float, default=1, help='Split')
argparser.add_argument('--epochs', type=int, default=20, help='Maximum epoch')
argparser.add_argument('--backbone', type=str, default='resnet18', help='Model')
argparser.add_argument('--Gaussiandrop', action='store_true', help='Gaussian')
argparser.add_argument('--batch_size', type=int, default=64, help='Batch size')
argparser.add_argument('--num_workers', type=int, default=2, help='Num workers')
argparser.add_argument('--nheads', type=int, default=8, help='Num workers')
argparser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
argparser.add_argument('--weights', type=list, default=[0.0730, 0.9269], help='Class weight')
argparser.add_argument('--eval', action='store_true', help='Running type')
argparser.add_argument('--val', action='store_true', help='Running type')
argparser.add_argument('--dist', action='store_true', help='Launch distributed training')
argparser.add_argument('--model', type=str, default='BaselineLSTM', help='Model architecture')
argparser.add_argument('--body', action='store_true',help='head2D use body')
argparser.add_argument('--rank', type=int, default=0, help='Rank id')
argparser.add_argument('--dsplit', type=float, default=0.1, help='Rank id')
argparser.add_argument('--csplit', type=float, default=0.1, help='Rank id')
argparser.add_argument('--start_rank', type=int, default=0, help='Start rank')
argparser.add_argument('--device_id', type=int, default=0, help='Device id')
argparser.add_argument('--world_size', type=int, help='Distributed world size')
argparser.add_argument('--init_method', type=str, help='Distributed init method')
argparser.add_argument('--backend', type=str, default='nccl', help='Distributed backend')

argparser.add_argument('--exp_path', type=str, default='output', help='Path to results')
argparser.add_argument('--train_path', type=str, default='output_train', help='Path to train_results')
argparser.add_argument('--checkpoint', type=str, help='Checkpoint to load')

argparser.add_argument('--num_encoder_layers', type=int, default=6, help='Encoder Layers')
argparser.add_argument('--num_decoder_layers', type=int, default=6, help='Decoder Layers')
argparser.add_argument('--dim_feedforward', type=int, default=2048, help='dim_feedforward')
argparser.add_argument('--dmodel', type=int, default=512, help='dim_feedforward')
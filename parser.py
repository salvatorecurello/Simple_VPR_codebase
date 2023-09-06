import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment
    parser.add_argument("--exp_name", type=str, default="default",
                        help="exp name")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint path")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="The number of places to use per iteration (one place is N images)")
    parser.add_argument("--img_per_place", type=int, default=4,
                        help="The effective batch size is (batch_size * img_per_place)")
    parser.add_argument("--min_img_per_place", type=int, default=4,
                        help="places with less than min_img_per_place are removed")
    parser.add_argument("--max_epochs", type=int, default=20,
                        help="stop when training reaches max_epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of processes to use for data loading / preprocessing")
    parser.add_argument("--loss", type=str, default='contrastive',
                        help="the loss type")
    parser.add_argument("--alpha", type=int, default=2,
                        help="alpha parameter of multisimilarity loss")
    parser.add_argument("--beta", type=int, default=50,
                        help="beta parameter of multisimilarity loss")
    parser.add_argument("--base", type=float, default=0.5,
                        help="base parameter of multisimilarity loss")
    parser.add_argument("--distance", type=str, default='cosinesimilarity',
                        help="distance used in multisimilarity loss")
    parser.add_argument("--miner", type=str, default='none',
                        help="type of miner")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="epsilon value of the multisimilarityminer")
    parser.add_argument("--optimizer", type=str, default='sgd',
                        help="choose the optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                        help="weight_decay for the optimizer")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum for the optimizer")
    parser.add_argument("--aggregator", type=str, default='avg', 
                        help="type of aggregation")
    parser.add_argument("--scheduler", type=str, default='none', 
                        help="scheduler")
    parser.add_argument("--tmax", type=int, default=5, 
                        help="value of T_max of CosineAnnealingLR")                    
                        
    # Architecture parameters
    parser.add_argument("--descriptors_dim", type=int, default=512,
                        help="dimensionality of the output descriptors")
    
    # Visualizations parameters
    parser.add_argument("--num_preds_to_save", type=int, default=0,
                        help="At the end of training, save N preds for each query. "
                        "Try with a small number like 3")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="When saving preds (if num_preds_to_save != 0) save only "
                        "preds for difficult queries, i.e. with uncorrect first prediction")

    # Paths parameters
    parser.add_argument("--train_path", type=str, default="data/gsv_xs/train",
                        help="path to train set")
    parser.add_argument("--val_path", type=str, default="data/sf_xs/val",
                        help="path to val set (must contain database and queries)")
    parser.add_argument("--test_path", type=str, default="data/sf_xs/test",
                        help="path to test set (must contain database and queries)")

    
    args = parser.parse_args()
    return args


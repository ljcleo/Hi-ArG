from train import TrainArg, parse_args, train

if __name__ == "__main__":
    args: TrainArg = parse_args()
    print(args)
    train(args, True)

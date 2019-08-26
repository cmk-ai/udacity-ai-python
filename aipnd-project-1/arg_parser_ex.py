import argparse

def show():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='directory', default="GOOD")
    parser.add_argument('--sum', dest='number')
    parser.add_argument('--rate', dest='rate', type=float, default=0.001)
    parser.add_argument('--gpu', dest='gpu', default='cuda')
    args = parser.parse_args()

    print(args.directory, type(args.directory))
    print(args.number)
    print(args.rate)
    print(args.gpu)
if __name__ == "__main__":
    show()

import argparse

def parser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--datapath', 
                        help='the path of the data', 
                        default='')
    return parser.parse_args()

def main():
    args = parser_args()

if __name__ == '__main__':
    main()
import argparse
import logging

logging.basicConfig(level=logging.INFO)

class BlockSplitter:
    def __init__(self, input_video: str, block_size: int):
        self.input_video = input_video
        self.block_size = block_size

    def split(self):
        # Implementation of block division
        logging.info(f'Splitting video {self.input_video} into blocks of size {self.block_size}.')
        pass  # Placeholder for processing logic

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Block Splitter for Mocha Pro.')
    parser.add_argument('input', help='Input video file.')
    parser.add_argument('size', type=int, help='Size of blocks for division.')
    args = parser.parse_args()

    splitter = BlockSplitter(args.input, args.size)
    splitter.split()

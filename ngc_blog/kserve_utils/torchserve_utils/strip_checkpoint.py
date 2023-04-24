import torch
import argparse
import argparse
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Strip and Prepare Checkpoint')
    parser.add_argument('--ckpt-path', default='trained_model.pth', help='path to checkpoint')
    parser.add_argument('--new-ckpt-name', default='trained_model_stripped.pth', help='name for stripped checkpoint')
    args = parser.parse_args()
    
    ckpt = torch.load(args.ckpt_path)
    torch.save(ckpt['model'],args.new_ckpt_name)
    
import os
import argparse
import sys
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find best score')
    parser.add_argument('--res_dir', type=str, default="DTUeval-python/eval_results/")
    args = parser.parse_args()
    fscores = dict()
    psnr_scores = dict()
    chamfer_dists = dict()

    for filename in os.listdir(args.res_dir):
        filename = args.res_dir + filename
        if filename.endswith("shading_eval.txt"):
            with open(filename, 'r') as f:
                text = f.read()
                psnr = re.search("psnr score: ([0-9]*[.])?[0-9]+", text).group(0)
                psnr_scores[filename] = float(psnr.split(": ")[-1])

        elif not filename.endswith("shading_eval.txt") and filename.endswith(".txt"):
            with open(filename, 'r') as f:
                text = f.read()
                chamfer_dist = re.search("overall: ([0-9]*[.])?[0-9]+", text).group(0)
                chamfer_dists[filename] = float(chamfer_dist.split(": ")[-1])
    
    sorted_psnr_scores = sorted(psnr_scores.items(), key=lambda x:x[1])
    sorted_chamfer_dists = sorted(chamfer_dists.items(), key=lambda x:x[1])

    print("min chamfer dist: %s %s %s %s %s" % (sorted_chamfer_dists[0],sorted_chamfer_dists[1],sorted_chamfer_dists[2],sorted_chamfer_dists[3],sorted_chamfer_dists[4]))
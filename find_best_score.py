import os
import argparse
import sys
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find best score')
    parser.add_argument('--res_dir', type=str, default="NeRFeval/eval_results/")
    args = parser.parse_args()
    fscores = dict()
    psnr_scores = dict()
    chamfer_dists = dict()

    for filename in os.listdir(args.res_dir):
        filename = args.res_dir + filename
        if filename.endswith("mesh_eval.txt"):
            with open(filename, 'r') as f:
                text = f.read()
                fscore = re.search("fscore: ([0-9]*[.])?[0-9]+", text).group(0)
                fscores[filename] = float(fscore.split(": ")[-1])

                chamfer_dist = re.search("loss: ([0-9]*[.])?[0-9]+", text).group(0)
                chamfer_dists[filename] = float(chamfer_dist.split(": ")[-1])

        elif filename.endswith("shading_eval.txt"):
            with open(filename, 'r') as f:
                text = f.read()
                psnr = re.search("psnr score: ([0-9]*[.])?[0-9]+", text).group(0)
                psnr_scores[filename] = float(psnr.split(": ")[-1])
    
    sorted_fscores = sorted(fscores.items(), key=lambda x:x[1])
    sorted_psnr_scores = sorted(psnr_scores.items(), key=lambda x:x[1])
    sorted_chamger_dists = sorted(chamfer_dists.items(), key=lambda x:x[1])

    print("max fscore: %s %s %s \nmax psnr score: %s %s %s\nmin chamfer dist: %s %s %s" % (sorted_fscores[-1], sorted_fscores[-2],sorted_fscores[-3],sorted_psnr_scores[-1], sorted_psnr_scores[-2],sorted_psnr_scores[-3],sorted_chamger_dists[0],sorted_chamger_dists[1],sorted_chamger_dists[2]))
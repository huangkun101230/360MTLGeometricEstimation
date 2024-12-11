import os
import torch
import numpy as np
import csv
#==========================
# Depth Prediction Metrics
#==========================

def compute_depth_metrics(gt, pred):
    """Computation of metrics between predicted and ground truth depths
    """

    gt[gt<0.1] = 0.1
    pred[pred<0.1] = 0.1
    gt[gt>10] = 10
    pred[pred>10] = 10


    ###########STEP 1: compute delta#######################
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    

    ##########STEP 2:compute mean error###################
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt) - torch.log10(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_ = torch.mean(torch.abs(gt - pred))

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    log10 = torch.mean(torch.abs(torch.log10(pred/gt)))

    return abs_, abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3


# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


class Depth_Evaluator(object):

    def __init__(self):
        # Error and Accuracy metric trackers
        self.metrics = {}
        self.metrics["err/abs_"] = AverageMeter()
        self.metrics["err/abs_rel"] = AverageMeter()
        self.metrics["err/sq_rel"] = AverageMeter()
        self.metrics["err/rms"] = AverageMeter()
        self.metrics["err/log_rms"] = AverageMeter()
        self.metrics["err/log10"] = AverageMeter()
        self.metrics["acc/a1"] = AverageMeter()
        self.metrics["acc/a2"] = AverageMeter()
        self.metrics["acc/a3"] = AverageMeter()

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        self.metrics["err/abs_"].reset()
        self.metrics["err/abs_rel"].reset()
        self.metrics["err/sq_rel"].reset()
        self.metrics["err/rms"].reset()
        self.metrics["err/log_rms"].reset()
        self.metrics["err/log10"].reset()
        self.metrics["acc/a1"].reset()
        self.metrics["acc/a2"].reset()
        self.metrics["acc/a3"].reset()

    def compute_eval_metrics(self, gt_depth, pred_depth, full_region=True):
        self.full_region = full_region
        """
        Computes metrics used to evaluate the model
        """
        N = gt_depth.shape[0]

        abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3 = \
            compute_depth_metrics(gt_depth, pred_depth)

        self.metrics["err/abs_"].update(abs_, N)
        self.metrics["err/abs_rel"].update(abs_rel, N)
        self.metrics["err/sq_rel"].update(sq_rel, N)
        self.metrics["err/rms"].update(rms, N)
        self.metrics["err/log_rms"].update(rms_log, N)
        self.metrics["err/log10"].update(log10, N)
        self.metrics["acc/a1"].update(a1, N)
        self.metrics["acc/a2"].update(a2, N)
        self.metrics["acc/a3"].update(a3, N)

    def print(self, dir=None):
        avg_metrics = []
        avg_metrics.append(self.metrics["err/abs_"].avg)
        avg_metrics.append(self.metrics["err/abs_rel"].avg)
        avg_metrics.append(self.metrics["err/sq_rel"].avg)
        avg_metrics.append(self.metrics["err/rms"].avg)
        avg_metrics.append(self.metrics["err/log_rms"].avg)
        avg_metrics.append(self.metrics["err/log10"].avg)
        avg_metrics.append(self.metrics["acc/a1"].avg)
        avg_metrics.append(self.metrics["acc/a2"].avg)
        avg_metrics.append(self.metrics["acc/a3"].avg)

        print("=====depth=====")
        print("\n  "+ ("{:>9} | " * 9).format("abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3"))
        print(("&  {: 8.5f} " * 9).format(*avg_metrics))

        if self.full_region:
            file_name = "depth_results_full_region.text"
        else:
            file_name = "depth_results_valid_region.text"

        if dir is not None:
            file = os.path.join(dir, file_name)
            with open(file, 'w') as f:
                print("\n  " + ("{:>9} | " * 9).format("abs_", "abs_rel", "sq_rel", "rms", "rms_log",
                                                      "log10", "a1", "a2", "a3"), file=f)
                print(("&  {: 8.5f} " * 9).format(*avg_metrics), file=f)
    
    def get_combined_err(self):
        err = np.array(self.metrics["err/abs_"].avg)+np.array(self.metrics["err/abs_rel"].avg)+np.array(self.metrics["err/sq_rel"].avg)+np.array(self.metrics["err/rms"].avg)+np.array(self.metrics["err/log_rms"].avg)
        return err
    
    def save_csv(self, folder_path, model_name):
        csv_file = folder_path+"evaluate_all_depth.csv"
        header = ["abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3"]
        avg_metrics = []
        avg_metrics.append(self.metrics["err/abs_"].avg)
        avg_metrics.append(self.metrics["err/abs_rel"].avg)
        avg_metrics.append(self.metrics["err/sq_rel"].avg)
        avg_metrics.append(self.metrics["err/rms"].avg)
        avg_metrics.append(self.metrics["err/log_rms"].avg)
        avg_metrics.append(self.metrics["err/log10"].avg)
        avg_metrics.append(self.metrics["acc/a1"].avg)
        avg_metrics.append(self.metrics["acc/a2"].avg)
        avg_metrics.append(self.metrics["acc/a3"].avg)
        
        with open(csv_file, mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_name])
            writer.writerow(header)
            writer.writerow(map(lambda x: str(x.numpy()), avg_metrics))
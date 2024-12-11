import os
import csv
import numpy as np

def normalise_tensor(tensor):
    array = tensor.detach().cpu().numpy()
    array = array.transpose(0,2,3,1)
    return array

def compute_normal_metrics(gt, pred, mask, radians, full_region):
    """Computation of metrics between predicted and ground truth normal
    ['Mean', 'Median', 'MSE', 'RMSE', 'error < 11.25°','error < 22.5°', 'error < 30°']
    """
    gt = normalise_tensor(gt)
    pred = normalise_tensor(pred)

    dot_products = np.sum(pred * gt, axis=3, keepdims=True)

    # Calculate the angle (in radians) between the two vectors using the arccosine
    angle_radians = np.arccos(np.clip(dot_products, -1.0, 1.0))

    if radians:
        E = angle_radians
    else:
        # Convert the angle from radians to degrees
        E = np.degrees(angle_radians)
        mask = mask.detach().cpu().numpy().transpose(0,2,3,1).astype(np.bool8)

        if full_region:
            E *= mask
        else:
            E = E[mask]


    return (np.mean(E), np.median(E),
             (np.mean(np.power(E,2))),
             np.sqrt(np.mean(np.power(E,2))),
             np.mean(E < 5) * 100,
             np.mean(E < 7.5) * 100,
             np.mean(E < 11.25) * 100,
             np.mean(E < 22.5 ) * 100,
             np.mean(E < 30   ) * 100)

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


class Norm_Evaluator_Valid(object):

    def __init__(self):
        # Error and Accuracy metric trackers
        self.metrics = {}
        self.metrics["err/mean"] = AverageMeter()
        self.metrics["err/median"] = AverageMeter()
        self.metrics["err/mse"] = AverageMeter()
        self.metrics["err/rmse"] = AverageMeter()
        self.metrics["acc/a1"] = AverageMeter()
        self.metrics["acc/a2"] = AverageMeter()
        self.metrics["acc/a3"] = AverageMeter()
        self.metrics["acc/a4"] = AverageMeter()
        self.metrics["acc/a5"] = AverageMeter()

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        self.metrics["err/mean"].reset()
        self.metrics["err/median"].reset()
        self.metrics["err/mse"].reset()
        self.metrics["err/rmse"].reset()
        self.metrics["acc/a1"].reset()
        self.metrics["acc/a2"].reset()
        self.metrics["acc/a3"].reset()
        self.metrics["acc/a4"].reset()
        self.metrics["acc/a5"].reset()

    def compute_eval_metrics(self, gt_norm, pred_norm, mask, radians=False, full_region=True):
        self.full_region = full_region
        """
        Computes metrics used to evaluate the model
        """
        N = gt_norm.shape[0]

        mean, median, mse, rmse, a1, a2, a3, a4, a5 = \
            compute_normal_metrics(gt_norm, pred_norm, mask, radians, full_region)

        self.metrics["err/mean"].update(mean, N)
        self.metrics["err/median"].update(median, N)
        self.metrics["err/mse"].update(mse, N)
        self.metrics["err/rmse"].update(rmse, N)
        self.metrics["acc/a1"].update(a1, N)
        self.metrics["acc/a2"].update(a2, N)
        self.metrics["acc/a3"].update(a3, N)
        self.metrics["acc/a4"].update(a4, N)
        self.metrics["acc/a5"].update(a5, N)

    def print(self, dir=None):
        avg_metrics = []
        avg_metrics.append(self.metrics["err/mean"].avg)
        avg_metrics.append(self.metrics["err/median"].avg)
        avg_metrics.append(self.metrics["err/mse"].avg)
        avg_metrics.append(self.metrics["err/rmse"].avg)
        avg_metrics.append(self.metrics["acc/a1"].avg)
        avg_metrics.append(self.metrics["acc/a2"].avg)
        avg_metrics.append(self.metrics["acc/a3"].avg)
        avg_metrics.append(self.metrics["acc/a4"].avg)
        avg_metrics.append(self.metrics["acc/a5"].avg)

        print("=====normal=====")
        print("\n  "+ ("{:>9} | " * 9).format("mean", "median", "mse", "rmse", "<5", "<7.5", "<11.25", "<22.5", "<30"))
        print(("& {: 9f} " * 9).format(*avg_metrics))

        if self.full_region:
            file_name = "normal_results_full_region.text"
        else:
            file_name = "normal_results_valid_region.text"

        if dir is not None:
            file = os.path.join(dir, file_name)
            with open(file, 'w') as f:
                print("\n  " + ("{:>9} | " * 9).format("mean", "median", "mse", "rmse", "<5", "<7.5", "<11.25", "<22.5", "<30"), file=f)
                print(("&{: 9f} " * 9).format(*avg_metrics), file=f)

    def get_combined_err(self):
        err = np.array(self.metrics["err/mean"].avg)+np.array(self.metrics["err/median"].avg)+np.array(self.metrics["err/rmse"].avg)
        return err
    
    def save_csv(self, folder_path, model_name):
        csv_file = folder_path+"evaluate_all_normal.csv"
        header = ["mean", "median", "mse", "rmse", "<5", "<7.5", "<11.25", "<22.5", "<30"]
        avg_metrics = []
        avg_metrics.append(self.metrics["err/mean"].avg)
        avg_metrics.append(self.metrics["err/median"].avg)
        avg_metrics.append(self.metrics["err/mse"].avg)
        avg_metrics.append(self.metrics["err/rmse"].avg)
        avg_metrics.append(self.metrics["acc/a1"].avg)
        avg_metrics.append(self.metrics["acc/a2"].avg)
        avg_metrics.append(self.metrics["acc/a3"].avg)
        avg_metrics.append(self.metrics["acc/a4"].avg)
        avg_metrics.append(self.metrics["acc/a5"].avg)
        
        with open(csv_file, mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_name])
            writer.writerow(header)
            writer.writerow(map(lambda x: str(x), avg_metrics))
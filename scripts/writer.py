import csv
import re
import torch

DELIMITER = ','

class Writer:
    def __init__(self, save_dir: str):
        # open files
        self.f_exp1 = open(f'{save_dir}/exp1.csv', 'w')
        self.f_exp2 = open(f'{save_dir}/exp2.csv', 'w')
        self.f_exp3 =  open(f'{save_dir}/exp3.csv', 'w')
        self.f_mask_t = open(f'{save_dir}/exp4a.csv', 'w')
        self.f_mask_v = open(f'{save_dir}/exp4b.csv', 'w')
        self.f_gen = open(f'{save_dir}/general.csv', 'w')
    
        # csv writer
        self.writer_exp1 = csv.writer(self.f_exp1, delimiter=DELIMITER)
        self.writer_exp2 = csv.writer(self.f_exp2, delimiter=DELIMITER)
        self.writer_exp3 = csv.writer(self.f_exp3, delimiter=DELIMITER)
        self.writer_exp4_mask_t = csv.writer(self.f_mask_t, delimiter=DELIMITER)
        self.writer_exp4_mask_v = csv.writer(self.f_mask_v, delimiter=DELIMITER)
        self.writer_gen = csv.writer(self.f_gen, delimiter=DELIMITER)
            
        # write headers
        self.writer_exp1.writerow(['test_name', 'esize', 'pval'])
        self.writer_exp2.writerow(['test_name', 'esize', 'pval'])
        self.writer_exp3.writerow(['test_name', 'esize_x', 'pval_x', 'esize_y', 'pval_y'])
        self.writer_exp4_mask_t.writerow(['test_name', 'esize', 'pval'])
        self.writer_exp4_mask_v.writerow(['test_name', 'esize', 'pval'])
        self.writer_gen.writerow(['test_name', 'X_AXonAY', 'X_BXonBY',
                                'Y_AXonAY', 'Y_BXonBY', 'X_AonB', 'Y_AonB',
                                'X_ABXonABY', 'Y_ABXonABY'])

    def format_test_name(self, test_name: str):
        test_name = re.sub('sent-|one_sentence|one_word', '', test_name)
        test_name = test_name[:-1] if test_name[-1] == '_' else test_name # strip trailing underscore
        return test_name

    def add_results_exp1(self, test_name: str, test_type: str, esize: float, pval: float):
        test_name = self.format_test_name(test_name)
        if isinstance(esize, torch.Tensor):
            esize = esize.item()
        self.writer_exp1.writerow([f'{test_name}:{test_type.strip("_")}', esize, f'{pval}'])

    def add_results_exp2(self, test_name: str, test_type: str, esize: float, pval: float):
        test_name = self.format_test_name(test_name)
        if isinstance(esize, torch.Tensor):
            esize = esize.item()
        self.writer_exp2.writerow([f'{test_name}:{test_type.strip("_")}', esize, f'{pval}'])

    def add_results_exp3(self, test_name: str, test_type: str, esize_x: float, pval_x: float, esize_y: float, pval_y: float):
        test_name = self.format_test_name(test_name)
        if isinstance(esize_x, torch.Tensor):
            esize_x = esize_x.item()
        if isinstance(esize_y, torch.Tensor):
            esize_y = esize_y.item()
        self.writer_exp3.writerow([f'{test_name}:{test_type.strip("_")}', esize_x, f'{pval_x}', esize_y, f'{pval_y}'])

    def add_results_exp4_mask_t(self, test_name: str, test_type: str, esize: float, pval: float):
        test_name = self.format_test_name(test_name)
        if isinstance(esize, torch.Tensor):
            esize = esize.item()
        self.writer_exp4_mask_t.writerow([f'{test_name}:mask_t_{test_type.strip("_")}', esize, f'{pval}'])

    def add_results_exp4_mask_v(self, test_name: str, test_type: str, esize: float, pval: float):
        test_name = self.format_test_name(test_name)
        if isinstance(esize, torch.Tensor):
            esize = esize.item()
        self.writer_exp4_mask_t.writerow([f'{test_name}:mask_v_{test_type.strip("_")}', esize, f'{pval}'])

    def flush(self):
        self.f_exp1.flush()
        self.f_exp2.flush()
        self.f_exp3.flush()
        self.f_mask_t.flush()
        self.f_mask_v.flush()
        self.f_gen.flush()
    
    def close(self):
        print('finished write to', self.f_exp1)
        self.f_exp1.close()
        self.f_exp2.close()
        self.f_exp3.close()
        self.f_mask_t.close()
        self.f_mask_v.close()
        self.f_gen.close()
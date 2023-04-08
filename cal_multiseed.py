import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='calulate the average perfromane via different seeds')
    parser.add_argument('path', help='source txt path')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    pAcc = []
    mIoU_S = []
    mIoU_U = []
    hIoU = []

    f = open(args.path, 'r')
    for line in f:
        if line.split('_').pop(0) != 'dino':
            head = line.split(':').pop(0)
            if head=='pAcc':
                pAcc.append(float(line.split(':').pop(-1).strip('\n')))
            elif head=='mIoU_S':
                mIoU_S.append(float(line.split(':').pop(-1).strip('\n')))
            elif head=='mIoU_U':
                mIoU_U.append(float(line.split(':').pop(-1).strip('\n')))
            elif head=='hIoU':
                hIoU.append(float(line.split(':').pop(-1).strip('\n')))
            else:
                assert AttributeError('Wrong saving!')
                
    mean_pAcc = np.array(pAcc).mean()
    mean_mIoU_S = np.array(mIoU_S).mean()
    mean_mIoU_U = np.array(mIoU_U).mean()
    mean_hIoU = np.array(hIoU).mean()
    f.close()


    f = open(args.path, 'a+')
    f.write('\n' + 'Total results:' + '\n')
    f.write('mean_pAcc: ' + str(mean_pAcc) +'\n')
    f.write('mean_mIoU_S: ' + str(mean_mIoU_S) +'\n')
    f.write('mean_mIoU_U: ' + str(mean_mIoU_U) +'\n')
    f.write('mean_hIoU: ' + str(mean_hIoU) +'\n')
    f.close()

if __name__ == '__main__':
    main()
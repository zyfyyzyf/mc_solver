# import time
# import subprocess
# starttime = time.time()
# status = subprocess.call(['timeout', '20', 'time', "./sharpSAT", '-decot', '1','-decow','100','-tmpdir','.','cs','32000', "/home/mc_zilla/data/raw_data/test_data/49.cnf"])
# endtime = time.time()
# dtime = endtime - starttime
# dtime = str(dtime)
# print(dtime[:4])  #显示到微秒
import joblib
import numpy as np 
test_result = [1]
test_time = [2]
np.savez('/home/mc_zilla/x.npz',test_result,test_time)

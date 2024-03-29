import IMU
import pickle
from time import sleep
import datetime

def write_data(path,data):
    with open(path,'wb+') as f:
        pickle.dump(data,f)

def load_data(path):
    with open(path,'rb+') as f:
        data=pickle.load(f)
    return data

def get_time():
    ISOTIMEFORMAT = '%d_%H_%M_%S'
    theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    return theTime


def collect_data(path,sample_rate=10):
    imu = IMU.IMUsensor()
    num=0
    data=[]
    while True:
        frame=imu.read_value()
        data.append(frame)
        num+=1
        if num%(sample_rate*20)==0:
            write_data(path,data)
        sleep(1/sample_rate)


if __name__=="__main__":
    PATH='/home/gix/TECHIN514/Shinmood/data_{}.pkl'.format(get_time())
    collect_data(PATH)





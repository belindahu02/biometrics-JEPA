from trainers import *
from pre_trainers import *
import numpy as np
import matplotlib.pyplot as plt

def plotspu(ft):
  scen = 1
  fet_extrct = pre_trainer(scen=scen, fet=6)
  
  if ft==0:
    model_name="musicid_scen"+str(scen)+"_simsiam"
  else:
    model_name="musicid_scen"+str(scen)+'_ft'+str(ft)+"_simsiam"
  
  variable_name="samples per user"
  variable=[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
  #variable=[15,20]
  acc=[]
  kappa=[]
  for el in variable:
    acc_temp=[]
    kappa_temp=[]
    for itr in range(10):
      test_acc, kappa_score = trainer(el, fet_extrct, scen, ft=ft)
      acc_temp.append(test_acc)
      kappa_temp.append(kappa_score)
    acc.append(acc_temp)
    kappa.append(kappa_temp)
  acc = np.array(acc)
  kappa = np.array(kappa)
  
  np.savez("graph_data/"+model_name+".npz", test_acc=acc, kappa_score=kappa)
  print(acc.shape)
  print(kappa.shape)
  
  kappa_max = np.max(kappa, axis=1)
  plt.figure(figsize=(12,8))
  plt.plot(variable,kappa_max, 'm', label=model_name)
  plt.title("kappa score vs "+variable_name)
  plt.xlabel(variable_name)
  plt.ylabel("kappa score")
  plt.legend()
  plt.show()
  if ft==0:
    plt.savefig('graphs/kappa_scen'+str(scen)+'.jpg')
  else:
    plt.savefig('graphs/kappa_scen'+str(scen)+'_ft'+str(ft)+'.jpg')
  plt.close()
  
  acc_max = np.max(acc, axis=1)
  plt.figure(figsize=(12,8))
  plt.plot(variable,acc_max, 'm', label=model_name)
  plt.title("test accuracy vs "+variable_name)
  plt.xlabel(variable_name)
  plt.ylabel("test acuracy")
  plt.legend()
  plt.show()
  if ft==0:
    plt.savefig('graphs/acc_scen'+str(scen)+'.jpg')
  else:
    plt.savefig('graphs/acc_scen'+str(scen)+'_ft'+str(ft)+'.jpg')
  plt.close()
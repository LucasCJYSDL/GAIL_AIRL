import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import copy


Sliding_Window = 60
Data_Length = 15000
plt.rcParams['font.family'] = "Times New Roman"

loss_data_1 = pd.read_csv("/home/jc/下载/train curve/csv/l/run-AIRL+SPARSE_trpo-tag-summary_1_cent_loss.scalar.summary_1.csv")
loss_data_2 = pd.read_csv("/home/jc/下载/train curve/csv/l/run-AIRL_trpo-tag-summary_1_cent_loss.scalar.summary_1.csv")
#loss_data_3 = pd.read_csv("/home/jc/下载/train curve/csv/l/run-GAIL_trpo-tag-summary_1_expert_loss.scalar.summary_1.csv")
#loss_data_4 = pd.read_csv("/home/jc/下载/train curve/csv/l/run-GAIL_trpo-tag-summary_1_generator_loss.scalar.summary_1.csv")

print (loss_data_1.Value.shape)
print (loss_data_2.Value.shape)
#print (loss_data_3.Value.shape)

loss_df_1 = pd.DataFrame(loss_data_1)
loss_df_2 = pd.DataFrame(loss_data_2)
#loss_df_3 = pd.DataFrame(loss_data_3)
#loss_df_4 = pd.DataFrame(loss_data_4)

term_1 = loss_df_1.Value.rolling(Sliding_Window, min_periods = 2).mean()
std_1 = loss_df_1.Value.rolling(Sliding_Window, min_periods = 2).std()
term_2 = loss_df_2.Value.rolling(80, min_periods = 2).mean()
std_2 = loss_df_2.Value.rolling(Sliding_Window, min_periods = 2).std()
#term_3 = loss_df_3.Value.rolling(40, min_periods = 2).mean()
#std_3 = loss_df_3.Value.rolling(Sliding_Window, min_periods = 2).std()
#term_4 = loss_df_4.Value.rolling(40, min_periods = 2).mean()

fig,ax1=plt.subplots()
#ax1.set_ylim([-15, 27])

'''axi_x = np.array([100, 700, 3000, 8000, 10000, 15000])

term_0 = np.array([24.2097, 24.2097, 24.2097, 24.2097, 24.2097, 24.2097])
term_1 = np.array([0.362973995, 11.31006047,17.1180197, 18.3814351, 19.88507975, 22.94129104])
std_1 = np.array([19.03100792, 17.2425488, 12.5877914, 13.81159643, 11.90778421, 8.788031299])
term_2 = np.array([7.632794611, 19.0699884, 21.1613617, 21.06124762, 22.85395969, 20.12598776])
std_2 = np.array([15.7301188, 12.21850468, 8.373650874, 8.398567088, 8.158568477, 10.3557806])
term_3 = np.array([12.85192431, 7.447492762, 16.21979121, 19.73882428, 20.12355165, 20.96380397])
std_3 = np.array([17.48067972, 14.22205073, 16.33204968, 12.30596956, 10.7267599, 11.40654357])'''

'''axi_x = np.array([100, 800, 1600, 9000, 14000, 15000])

term_0 = np.array([1,1,1,1,1,1])
term_1 = np.array([0.22,0.56,0.94,1,1,1])
#std_1 = np.array([19.03100792, 17.2425488, 12.5877914, 13.81159643, 11.90778421, 8.788031299])
term_2 = np.array([0.3,0.94,1,1,1,1])
#std_2 = np.array([15.7301188, 12.21850468, 8.373650874, 8.398567088, 8.158568477, 10.3557806])
term_3 = np.array([0.46,0.4,0.64,0.88,0.94,0.66])
#std_3 = np.array([17.48067972, 14.22205073, 16.33204968, 12.30596956, 10.7267599, 11.40654357])'''


'''axi_x = np.array([100, 500, 1600, 7000, 10000, 15000])

term_0 = np.array([58.18,58.18,58.18,58.18,58.18,58.18])
term_1 = np.array([68.96774194,64.95833333,57.34042553,58.54,57.38,56.38])
std_1 = np.array([12.19858234,12.35407884,11.17991473,13.18364138,8.529689326,10.3535308])
term_2 = np.array([68,57.41666667,55.68,54.97959184,58.42,56.46])
std_2 = np.array([17.4560673,12.91505366,9.321888221,12.73992567,7.494237786,10.13944772])
term_3 = np.array([65.14285714,59.17857143,60.84375,59.32653061,59.18181818,58.27272727])
std_3 = np.array([21.81368228,19.14055823,17.44295949,12.3211909,10.88216374,11.72739018,])'''

'''axi_x = np.array([100, 1800, 4000, 6000, 10000, 14000])

term_0 = np.array([67.74,67.74,67.74,67.74,67.74,67.74])
term_1 = np.array([118.5,72.7,71.82,66.54,72.5,84.58])
std_1 = np.array([43.56982901,25.05613697,19.2100911,25.1485268,25.52978652,37.34064274])
term_2 = np.array([88.44,65.1,64.64,71.36,67.96,65.4])
std_2 = np.array([64.03254173,22.1542321,30.16007958,23.98646285,17.0598476,14.95593528])
term_3 = np.array([108.9,94.64,77.24,68.5,96.72,71.38])
std_3 = np.array([55.9153825,55.85329355,20.7418032,37.07843039,39.65251064,30.94439529])'''

axi_x = loss_df_1.Step

#term_0 = np.array([24.21]*1000)

#l0, = ax1.plot(axi_x,term_0,color = "black",alpha =1, linestyle='--', linewidth=2)

l1, = ax1.plot(axi_x,term_1,color = "red",alpha =1, linewidth=2)
#plt.fill_between(axi_x, term_1 + std_1, term_1 - std_1, color="yellow", alpha=0.3)

l2, = ax1.plot(axi_x, term_2, color = "orange",alpha =1, linewidth=2)
#plt.fill_between(axi_x, term_2 + std_2, term_2 - std_2, color="red", alpha=0.3)

#l3, = ax1.plot(axi_x, (term_3 + term_4) / 2, color = "blue",alpha =1, linewidth=2)
#plt.fill_between(axi_x, term_3 + std_3, term_3 - std_3, color="blue", alpha=0.3)

#ax1.set_xlabel('train_step')
#ax1.set_ylabel('accumulated_reward')



#plt.legend(handles = [l1,l2,l3,l4], labels =["loss","expert_reward","policy_weighted_reward","policy_unweighted_reward"])
#plt.legend(handles = [l1, l2, l3], labels =["AIRL+SPARSE","AIRL", "GAIL"],loc = 1)
plt.legend(handles = [l1, l2], labels =["AIRL+SPARSE","AIRL"],loc = 1)
ax1.set_title('Discriminator Loss',fontsize=12,color='black')

plt.show()

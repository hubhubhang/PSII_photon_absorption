# this code plot the data generated from "Josh_fig2.py"

import os
import numpy as np
import matplotlib.pylab as plt

#======================================[0] Content ========================================
#[1] Load data from npy file
#[2] Plotting
#    2.1 Axis range, 2.2 Labels, 2.3 Remarks

#=================================[1] Load data from npy file=================================

list_all = os.listdir(".")                                     # return a list containing the names of the entries in the current directory
list_npy = [list_all[i] for i in range(len(list_all)) if list_all[i].endswith('1.npy')] 

for i in range(len(list_npy)): 
	f = open(list_npy[i], "rb")                                # load Numpy arrays from binary files

	load_initial_time1 = np.load(f)
	load_dt1 = np.load(f)
	load_time1 = np.load(f)
	load_precise_times1 = np.load(f)

	load_op_array_list1 = np.load(f)
	load_P_exe1_1 = np.load(f)
	load_P_exe2_1 = np.load(f)
	   
	load_Pmax1 = np.load(f)
	load_runtime1 = np.load(f)

	load_photon_no = np.load(f) 
	load_capital_gamma = np.load(f)
	load_fre_band = np.load(f)
	load_t_peak = np.load(f)
	load_t_limits_multiple = np.load(f)

	f.close()


#=================================[2] Plotting=================================-
#2.1 Axis range
#2.2 Labels
#2.3 Remarks

fig, ax1 = plt.subplots()
ax1.plot(load_precise_times1*(load_capital_gamma), np.real(load_P_exe1_1), 'g', label = '|e>')
ax1.plot(load_precise_times1*(load_capital_gamma), np.real(load_P_exe2_1), 'k', label = '|g>')

#------------------------[2.1] Axis range-------------------------------
x_axis_min = 0                
x_axis_max = 14         
y_axis_min = 0               
y_axis_max = 1              
ax1.axis((x_axis_min, x_axis_max, y_axis_min, y_axis_max )) 

#------------------------[2.2] Labels------------------------------------
plt.title("Single-photon Fock state", y = 1.05)
#plt.title("Two-photon Fock state", y = 1.05)
#plt.title("Equal superposition of single- and two-photon Fock states", y = 1.05)
ax1.set_xlabel('Time [1/' + r"$\Gamma$" +']')
ax1.set_ylabel('Excitation Probability  ' + r'$\mathbb{P}_{e}$')  
plt.subplots_adjust(left=0.17, right=0.85, top=0.9, bottom=0.23)
ax1.legend(bbox_to_anchor=(0.8, 0.97), loc=2, borderaxespad=0., fontsize = 10)

#------------------------[2.3] Remarks----------------------------------
y_axis = y_axis_min - (y_axis_max-y_axis_min)/3.6
y_axis_tuning = y_axis_max/25.

ax1.text(-2, y_axis+y_axis_tuning,\
	   'Remarks: 1.Simulated by RK4 with timestep = (1/' + r"$\Gamma$"  +')' + r"$\times$" + '(' + '1e-3' + ')'  + '\n' +\
'                2.Photon bath: N=' + str(int(load_photon_no)) + ', ' + r"$\Gamma$" + '=' + str(load_capital_gamma) +\
								  '(1/ps)' + ', fre_band=' +  str(load_fre_band/load_capital_gamma) + r"$\Gamma$"  +\
								  ', SD=' + str(int(load_t_limits_multiple)) + '(1/' + r"$\Gamma$)" + ', ' + r"$t_{peak}$"  + '=' + str(int(load_t_peak))  + '\n' +\
'                3.' + r'$\mathbb{P}_{e}^{max}$'+ '=' + str(np.max(np.real(load_P_exe1_1)))           , fontsize = 9.5)
#------------------------------------------------------------------------

plt.savefig('Josh fig 2(a) ().pdf')

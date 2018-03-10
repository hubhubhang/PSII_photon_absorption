# this code regnerates fig2a (Fock-state Master Eqns for a two-level atom interacting with a Gaussian wave packet) in the article
# "N-photon wave packets interacting with an arbitrary quantum system" (Baragiola 2012)

import os
import sys
import math
import numpy as np
from sklearn.utils.extmath import fast_dot

from datetime import datetime
start=datetime.now()

#==================================================================================================
#========================================USER MANUAL===============================================
#==================================================================================================

# 1. Input parameters below (below are the most common parameters to vary).
# 2. Change the expression for "combined_rho" in section [6] RK4, 
#    depending on whether the initial field state is an N-photon Fock state (default) or superposition of Fock states.
#    See eqn(44) of Josh's paper.
# 3. No need to modify anything after the END of USER MANUAL unless you want to do something else.
# 4. Run this python code, then a data file in .npy format will be generated. 
#    Run "Josh_fig2_plot.py" to plot the data, a pdf file will be generated.
#    Check and modify the labels and title of plot as needed.

#-------------------------------------Parameters Input----------------------------------------------
#Photon pulse-------------------------------- 
photon_no = 1.
fock_state_level = 1                            # integer that must be >= photon_no                 
capital_gamma = 1.
fre_band = capital_gamma * 1.46  
t_peak = 5.                                     # peak of the Gaussian pulse (in unit of 1/capital_gamma)
t_limits_multiple = 3.0                         # No. of standard deviation of the Gaussian pulse included
#initial density matrix----------------------
rho = np.array([[0., 0.],                        
                [0., 1.]])
#simulation time-----------------------------
initial_time = (0.)*(1./capital_gamma)
final_time = (12.)*(1./capital_gamma) 
N_data_pt = 1000 
dt = (1./capital_gamma) / (1e3)

#==================================================================================================
#========================================END of USER MANUAL========================================
#==================================================================================================


#======================================[0] Content ================================================
#[1] Constant
#[2] Calculation Tools
#[3] Parameters setting
#    3.1 Hamiltonian, 3.2 Photon pulse, 3.3 Gaussian envelope parameter, 3.4 Optimization constant 
#[4] Initial condition
#    4.1 Density matrix, 4.2 Time, 4.3 Observables
#[5] Equations
#[6] RK4 
#[7] Save output to npy.file

#=================================[1] Constant =========================================

h_bar = 1.054571800*(1e-34)        # Reduced Plank constant[Js]
k_B = 1.38064852*(1e-23)           # Boltzmann constant [J/K]
vac_per = 8.854187817620*(1e-12)   # vacuum permittivity [C^2/(N*m^2)] (from wiki)
lightspeed = 2.99792458*(1e8)      # speed of light in [m/s]
pi = 3.141592653589793 

J2cm = 5.034117213*(1e22)          # multiply this constant to convert from J to cm^-1   [cm-1/J]
s2ps = (1e12)                      # multiply this constant to convert from s to ps      [ps/s]
cm2Hz = 2.99792458*(1e10)          # multiply this constant to convert from cm^-1 to Hz  [Hz/cm-1]
D2Cm = 3.33564095*(1e-30)          # multiply this constant to convert from Debye to Cm  

#--------------------------------------------------------------------------------------
h_bar_in_cm_ps = h_bar*J2cm*s2ps                                      # [cm^-1 *ps]
k_B_in_cm_K = k_B*J2cm                                                # [cm^-1 / K]
inverse_h_bar_in_cm_ps = 1./(h_bar_in_cm_ps)                          # 1/(h_bar)
inverse_h_bar_in_cm_ps_sq = 1./(h_bar_in_cm_ps*h_bar_in_cm_ps)        # 1/(h_bar^2)

#-----------------------------------Pauli operators------------------------------------
X = np.array([[0.,1.],[1.,0.]])
Y = np.array([[0.,-1.j],[1.j,0.]])
Z = np.array([[1.,0.],[0.,-1.]])


#=================================[2] Calculation Tools ================================
def commutator(A,B):
    return fast_dot(A,B) - fast_dot(B,A)
 
def anticommutator(A,B):
    return fast_dot(A,B) + fast_dot(B,A)

def Dissipator(L, rho):
    L_dagger = np.conjugate(L.T)
    return fast_dot(fast_dot(L, rho), L_dagger) - 0.5*anticommutator(fast_dot(L_dagger, L), rho)


#=================================[3] Parameters setting ================================
#3.1 Hamiltonian
#3.2 Photon pulse
#3.3 Gaussian envelope parameter
#3.4 Optimization constant 

#-------------------------------------[3.1] Hamiltonian (monomer)-------------------------------------

N = 2                                           # No. of state                                                                                        
E_1 = 12800.                                    # Monomer's energy
H_s = np.array([[E_1, 0.], [0., 0.]])           # system Hamiltonian (monomer)
H_s_I = 0.                                      # system Hamiltonian in RWA

#-------------------------------------[3.2] Photon bath----------------------------------------------- 

photon_no = photon_no                 
capital_gamma = capital_gamma
fre_band = fre_band               

state_e = np.array([[1.],[0.]])     # |e>
state_g = np.array([[0.],[1.]])     # |g> 

L_1 = np.sqrt(capital_gamma)*fast_dot(state_g, state_e.T)  # deexcitation by photon emission (monomer)
L_1_dagger = np.conjugate(L_1.T)

#-------------------------------------[3.3] Gaussian pulse parameter -------------------------
 
t_peak = t_peak                                        # peak of the pulse (in unit of 1/capital_gamma)
t_limits_multiple = t_limits_multiple                  # No. of standard deviation of the pulse included
sigma = np.sqrt(2.)/ fre_band                          # standard deviation of the pulse
t_init = t_peak - t_limits_multiple * sigma            # time that the pulse comes in
t_final = t_peak + t_limits_multiple * sigma    

print('t_init=', t_init*capital_gamma, "1/capital_gamma")
print('t_final=', t_final*capital_gamma, "1/capital_gamma")

#-------------------------------------[3.4] Optimization constant --------------------------------
sq_fre_band = fre_band*fre_band
amplitude = np.sqrt(np.sqrt((sq_fre_band/(2.0*pi))))


#=================================[4] Initial condition ================================
#4.1 Density matrix
#4.2 Time
#4.3 Observables

#---------------------------------[4.1] Density matrix ---------------------------------
fock_state_level = fock_state_level        # 00, 01, 10, and 11; 10 is hermitian conjugate of 01
if fock_state_level < photon_no:
    print('error, please increase fock_state_level!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' +
          '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    sys.exit()
fock_state_level_plus_one = fock_state_level + 1 
fock_state_lv_list = range(fock_state_level+1)[1:]
print("fock_state_lv_list=", fock_state_lv_list)

op_array_list = np.zeros((fock_state_level_plus_one, fock_state_level_plus_one, N, N), dtype = complex)
rho = rho 
for i in range(fock_state_level_plus_one):
    op_array_list[i,i] = rho

pre_result = np.zeros((fock_state_level_plus_one, fock_state_level_plus_one, N, N), dtype = complex) 

#----------------------------------------[4.2] time ---------------------------------------------------------
initial_time = initial_time
final_time = final_time 

print('starting time=', initial_time*capital_gamma, "1/capital_gamma")

N_data_pt = N_data_pt
time = initial_time  
times = np.linspace(initial_time, final_time, N_data_pt)      # times to iterate
precise_times = np.zeros(N_data_pt)                           # times to record

dt = dt

#----------------------------------------[4.3] Observables ----------------------------------------------------

P_exe1 = np.zeros(N_data_pt, dtype = complex)
P_exe2 = np.zeros(N_data_pt, dtype = complex)


#=================================[5] Equations ================================
#below functions implement eqn(21) of Josh's paper
def dwholedt(t, op_array_list_local, result=pre_result):   # whole = op_array_list_local[i]

    if t_init <= t <= t_final:
        gaussian = amplitude*np.exp(-0.25*sq_fre_band*(t-t_peak)*(t-t_peak))             # Gaussian function  
    else:
        gaussian = 0.

#below is in interaction pic (H_s_I removed since already removed in Josh's eqn)
    for a in fock_state_lv_list: 
        for b in fock_state_lv_list: 
            op_array_list_local_ab = op_array_list_local[a,b]
            fock_state_term1 = np.sqrt(a)*gaussian*commutator(op_array_list_local[a-1,b], L_1_dagger) \
                             + np.sqrt(b)*np.conjugate(gaussian)*commutator(L_1, op_array_list_local[a,b-1])
            result[a,b] = fock_state_term1 + Dissipator(L_1, op_array_list_local_ab)

    a=0
    for b in fock_state_lv_list:
        op_array_list_local_ab = op_array_list_local[a,b]            
        fock_state_term1 = np.sqrt(b)*np.conjugate(gaussian)*commutator(L_1, op_array_list_local[a,b-1])
        result[a,b] = fock_state_term1 + Dissipator(L_1, op_array_list_local_ab)

    b=0
    for a in fock_state_lv_list:
        op_array_list_local_ab = op_array_list_local[a,b]            
        fock_state_term1 = np.sqrt(a)*gaussian*commutator(op_array_list_local[a-1,b], L_1_dagger) 
        result[a,b] = fock_state_term1 + Dissipator(L_1, op_array_list_local_ab) 

    a=0
    b=0
    op_array_list_local_ab = op_array_list_local[a,b]  
    result[a,b] = Dissipator(L_1, op_array_list_local_ab)

    return result

#=================================[6] RK4 ================================ 

j = 0

while j < N_data_pt:
    if time >= times[j]:
# see Josh's paper eqns (28) & (44)
#        combined_rho = op_array_list[1,1]                                 # for N=1 case 
#        combined_rho = op_array_list[2,2]                                 # for N=2 case 
#        combined_rho = 0.5*op_array_list[1,1] + 0.5*op_array_list[2,2] \
#                     + 0.5*op_array_list[1,2] + 0.5*op_array_list[2,1]    # for equal superposition of N=1 and N=2 fock states
        combined_rho = op_array_list[int(photon_no), int(photon_no)]       # for N = any integer case

        P_exe1[j] = combined_rho[0,0]
        P_exe2[j] = combined_rho[1,1]      
        precise_times[j] = time

        print('j=', j)
        print('P_exe1=', P_exe1[j])
        print('P_exe2=', P_exe2[j])        
        print('time=', precise_times[j]*capital_gamma, "1/Capital_gamma")        
        j = j + 1

    k1 = dt*dwholedt(time, op_array_list)
    k2 = dt*dwholedt((time+ 0.5*dt), (op_array_list+0.5*k1))
    k3 = dt*dwholedt((time+ 0.5*dt), (op_array_list+0.5*k2))
    k4 = dt*dwholedt((time+dt), (op_array_list+k3))
    op_array_list = op_array_list + (k1 + 2.*(k2 + k3) + k4 )/6.
    time = time + dt


Pmax = np.max(P_exe1)
print('Pmax=', Pmax)
runtime = datetime.now()-start
print (runtime)

#=================================[7] save output to npy.file ================================

save_result = open("result_" + os.path.basename(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "_S1.npy", "wb")

np.save(save_result, initial_time)
np.save(save_result, dt)
np.save(save_result, time)
np.save(save_result, precise_times)

np.save(save_result, op_array_list)
np.save(save_result, P_exe1)
np.save(save_result, P_exe2)

np.save(save_result, Pmax)
np.save(save_result, runtime)

np.save(save_result, photon_no)
np.save(save_result, capital_gamma)
np.save(save_result, fre_band)
np.save(save_result, t_peak)
np.save(save_result, t_limits_multiple)

save_result.close()
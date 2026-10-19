import numpy as np
import json
import scipy.special as sp
import os
import argparse
import matplotlib.pyplot as plt

def load_probe_data(n, simDir, emptyDir,spaceStep,n_freqs):
    Location = np.loadtxt(f'{simDir}/multipole_monitor/OUTPUTEfield_Tracker_' + str(n) + '.txt',max_rows = 1)
    x,y,z = Location * spaceStep
    r = np.sqrt(x*x+y*y+z*z); t = np.arccos(z/(np.sqrt(x*x+y*y+z*z))); p = np.arctan2(y,x)
    if p < 0:
        p += 2*np.pi
    A = np.loadtxt(f'{simDir}/multipole_monitor/OUTPUTEfield_Tracker_' + str(n) + '.txt',skiprows=1)
    A0 = np.loadtxt(f'{emptyDir}/multipole_monitor/OUTPUTEfield_Tracker_' + str(n) + '.txt',skiprows=1)
    wl = A[:,0]
    Ex = (A[:,1] + 1j*A[:,2]) - (A0[:,1] + 1j*A0[:,2]); Hx = (A[:,7]  + 1j*A[:,8] ) - (A0[:,7]  + 1j*A0[:,8] )
    Ey = (A[:,3] + 1j*A[:,4]) - (A0[:,3] + 1j*A0[:,4]); Hy = (A[:,9]  + 1j*A[:,10]) - (A0[:,9]  + 1j*A0[:,10])
    Ez = (A[:,5] + 1j*A[:,6]) - (A0[:,5] + 1j*A0[:,6]); Hz = (A[:,11] + 1j*A[:,12]) - (A0[:,11] + 1j*A0[:,12])
    r_dot_Er = (x*Ex + y*Ey + z*Ez)
    r_dot_Hr = (x*Hx + y*Hy + z*Hz)
    probe_data = {
        'wavelength': A[:,0],'x':x,'y':y,'z':z,'radius':r,'theta':t,'phi':p, 
        'Ex':Ex, 'Ey':Ey, 'Ez':Ez, 'r.Er':r_dot_Er,
        'Hx':Hx, 'Hy':Hy, 'Hz':Hz, 'r.Hr':r_dot_Hr,}
    return probe_data


def Hankel(x,n):
    OUT = np.zeros(len(x),dtype=np.complex128)
    for i in range(len(x)):
        OUT[i] = sp.spherical_jn(n,x[i]) + 1j*sp.spherical_yn(n,x[i])
    return OUT


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description ='Electrocmagnetic Field Multipole Extraction')
    parser.add_argument('--Simulation_Directory', '-simDir', type=str, required=False, default='.')
    parser.add_argument('--Empty_Simulation_Directory', '-emptyDir', type=str, required=False, default=None)
    parser.add_argument('--Output_Directory', '-outDir', type=str, required=False, default=None)
    parser.add_argument('--maximum_multipole_order', '-lmax', type=int, required=False, default=2)
    parser.add_argument('--multipole_degree_case', '-mcase', type=int, required=False, default=-1)

    #############################################################
    #           Gather Argparse Keyword Arguments               #
    #############################################################
    kwargs = parser.parse_args().__dict__                       #
    simDir = os.path.abspath(kwargs['Simulation_Directory'])    #
    if kwargs['Empty_Simulation_Directory'] == None:            #
        emptyDir = f'{simDir}/EmptySim'                         #
    else: 
        emptyDir = kwargs['Empty_Simulation_Directory']
    if kwargs['Output_Directory'] == None:                      #
        outDir = f'{simDir}/MultipoleData'                      #
    #determine the multipole terms to be collected              #
    l_max = kwargs['maximum_multipole_order']                   #
    m_case = kwargs['multipole_degree_case']                    #
    multipole_terms = []                                        #
    for l in range(1,l_max+1):                                  #
        if m_case <= 0:                                         #
            for m in range(l*m_case,l+1):                       #
                multipole_terms.append((l,m))                   #
        else:                                                   #
            for m in range(m_case,l+1):                         #
                multipole_terms.append((l,m))                   #  
    #############################################################

    #####################################################################
    #           Print Processed Keyword Arguments to Terminal           #       
    #####################################################################
    print(f'\nSimulation Directory: {simDir}')                          #
    print(f'Empty Simulation Directory: {emptyDir}')                    #
    print(f'Output Data Directory: {outDir}')                           #
    print(f'\nExtracting {len(multipole_terms)} Multipole Terms')       #
    print(f'Multipole Terms to Be Extracted (l,m): {multipole_terms}')  #
    #####################################################################

    #############################################################
    #                       Read INI files                      #
    #############################################################
    antenna_types = {                                           #
        '0': 'Empty', '1':'Sphere',                             #
        '12':'Cylinder', '1000':'Custom'}                       #
    with open(f'{simDir}/INIDEF/pphinfoini.json', 'r') as file: #
        ini = json.load(file)                                   #
        dx,dy,dz = ini['Space Step']                            #
        r, n_theta, n_phi = ini['Multipoles']                   #
        ant_type = antenna_types[str(ini['Antenna Type'])]      #
        if 'Polarization Angle' in ini:                         #
            pol = ini['Polarization Angle']                     #
        else:                                                   #
            pol = 90                                            #
        n_freqs = ini['Number of Wavelengths']                  #
    n_probes = n_theta*n_phi                                    #
    #############################################################

    #############################################################
    #       Print Multipole Monitor Details from INI            #
    #############################################################
    print(f'\n{n_probes} Multipole Monitor Probes')             #
    print(f'Multipole Monitor Probe Radius: {r*dy:.0e} meters') #
    #############################################################

    #############################################################
    #                   TODO: Read Domain Files                 #
    #############################################################

    #############################################################
    #                   Read Q Coefficient Data                 #
    #############################################################
    qFile = f'{simDir}/OUTPUT/ART/Qcoeff_double_{pol}pol.txt'   #
    Q_wl, Q_sca, Q_abs, Q_ext = np.loadtxt(qFile, unpack=True)  #
    #############################################################


    #########################################################################################
    #                           Extract Multipole Coefficients                              #
    #########################################################################################

    try:
        os.mkdir(outDir)
    except FileExistsError:
        ""
    try:
        os.mkdir(f'{outDir}/plots')
    except FileExistsError:
        ""
    try:
        os.mkdir(f'{outDir}/data')
    except FileExistsError:
        ""


    phi = np.linspace(0,2*np.pi,n_phi)
    theta = np.linspace(0,np.pi,n_theta)
    dphi = phi[1] - phi[0]
    dtheta = theta[1] - theta[0]
    z0 = 376.73
    multipole_data = np.zeros((len(multipole_terms), n_freqs, 5))
    multipole_power = np.zeros((len(multipole_terms), n_freqs, 3))
    total_power = np.zeros((n_freqs, 2))
    for n in range(len(multipole_terms)):
        l = multipole_terms[n][0]
        m = multipole_terms[n][1]
        print(f'{l},{m}')
        aE = 0
        aH = 0
        radius = np.zeros((n_probes),dtype=np.float64)
        check = 0
        for i in range(n_probes):
            probe_data = load_probe_data(i,simDir,emptyDir,dx,n_freqs)
            radius[i] = probe_data['radius']
            wavelength = probe_data['wavelength']
            Y = sp.sph_harm_y(l,m,probe_data['theta'],probe_data['phi'])
            Y_conj = np.conjugate(Y)
            aE += Y_conj*probe_data['r.Er']*np.sin(probe_data['theta'])*dphi*dtheta
            aH += Y_conj*probe_data['r.Hr']*np.sin(probe_data['theta'])*dphi*dtheta
            check += Y_conj*Y*np.sin(probe_data['theta'])*dphi*dtheta
            
        print(check)
        mean_radius = np.mean(radius)
        wavenumber = np.pi*2.0/wavelength
        h = Hankel(wavenumber*mean_radius,l)
        aE = -aE*wavenumber/np.sqrt(l*(l+1))
        aE = aE/(z0*h)
        aH = aH*wavenumber/np.sqrt(l*(l+1))
        aH = aH/(h)

        multipole_data[n, :, 0] = wavelength
        multipole_data[n, :, 1] = np.real(aE)
        multipole_data[n, :, 2] = np.imag(aE)
        multipole_data[n, :, 3] = np.real(aH)
        multipole_data[n, :, 4] = np.imag(aH)
        multipole_power[n, :, 0] = wavelength
        multipole_power[n, :, 1] = (z0/(2*wavenumber**2))*np.abs(aE)**2
        multipole_power[n, :, 2] = (z0/(2*wavenumber**2))*np.abs(aH)**2
        total_power[:, 0] = wavelength
        total_power[:, 1] += multipole_power[n, :, 1] + multipole_power[n, :, 2]

        np.savetxt(f'{outDir}/data/multipole_{l}{m}.txt', multipole_data[n, :, :])
        np.savetxt(f'{outDir}/data/power_{l}{m}.txt', multipole_power[n, :, :])

        plt.plot(np.real(multipole_data[n, :, 0])*1E9,np.abs(aE), label='$|aE|$')
        plt.plot(np.real(multipole_data[n, :, 0])*1E9,np.real(aE), label='$re(aE)$')
        plt.plot(np.real(multipole_data[n, :, 0])*1E9,np.imag(aE), label='$im(aE)$')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel(f"Multipole Coefficient {l}, {m} (A/m)")
        plt.legend()
        plt.title(f"Electric Multipole Coefficient {l}, {m}")
        plt.tight_layout()
        plt.savefig(f'{outDir}/plots/aE{l}{m}.png')
        plt.close()
        plt.plot(np.real(multipole_data[n, :, 0])*1E9,np.abs(aH), label='$|aH|$')
        plt.plot(np.real(multipole_data[n, :, 0])*1E9,np.real(aH), label='$re(aH)$')
        plt.plot(np.real(multipole_data[n, :, 0])*1E9,np.imag(aH), label='$im(aH)$')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel(f"Multipole Coefficient {l}, {m} (A/m)")
        plt.legend()
        plt.title(f"Magnetic Multipole Coefficient {l}, {m}")
        plt.tight_layout()
        plt.savefig(f'{outDir}/plots/aH{l}{m}.png')
        plt.close()

    np.savetxt(f'{outDir}/data/totalpower.txt', total_power)
    plt.title(f"Total Radiated Power")
    plt.plot(total_power[:,0]*1E9, total_power[:,1])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Total Radiated Power (W)")
    plt.savefig(f'{outDir}/plots/totalpower.png')
    plt.tight_layout()
    plt.close()
    ##########################################################################################
        
        







    
        



    
        



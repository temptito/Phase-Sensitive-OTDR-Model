import numpy as np


class otdr():
    
    def __init__(self):
        self.test_params = {
                    'fiber': {'L': 1000, 'n_rc': 3, 'z_rp': [], 'K_rp': [-40]},
                    'receiver': {'Ladc': 1, 'long_time': 1},
                    'pulse': {'tau': 200e-9, 'fp': 1e3, 'nyu_drift': 0.1e6, 
                              'lorentz_linewidth': 1e3, 'chirp_value': 0, 'dw': 80e6},
                    'PZT': {'K_mod': 0, 'F_mod': 0, 'Z_pzt': 500, 'len_pzt': 30}
                    }
        
        self.alpha = 0.0410517 * 1e-3 # attenuation, 1/m
        self.c = 300000000 # light speed, m/s
        self.n0 = 1.5      # refractive index
        lamda = 1550e-9 # wavelength, m
        kv = 2*np.pi/lamda # wavenumber, 1/m  
        self.w0 = kv * self.c / self.n0 # carrier frequency, rad/s
    
    def set_params(self, params):
        """
        setting parameters of 
        fiber: L - fiber length in m, n_rc - number of RC per meter in 1/m;
        receiver: Ladc - sampling frequency of receiver in m, T - time of capturing data in s;
        pulse: tau - pulse duration in s, fp - probing frequency in Hz, nyu_drift - linear carrier frequency drift in Hz/s,
        lorentz_linewidth - in Hz, dw - pulse modulation in Hz, 
		chirp_value - pulse chirp range in Hz(starts from dw to dw + chirp_value);
        PZT: K_mod - coefficient in rad, F_mod - frequency modualtion in Hz,  
        Z_pzt - PZT position in m, len_pzt - PZT length.
        """
        
        """set_fiber"""
        L = params['fiber']['L']
        n_rc = params['fiber']['n_rc']
        np.random.seed(1) # fixing random RC distribution
        self.z_rc = np.sort(np.random.rand(int(n_rc*L)))*L # RC's coordinates along fiber

        n_rc_ns = n_rc * self.c / 1e9 / self.n0                                         # number RC per nanosec
        K_rc_ns = -80                                                                 #reflect on RC per nanosec
        K_rc = 10**(0.1*K_rc_ns) / n_rc_ns
        K_rc = np.ones(len(self.z_rc)) * np.sqrt(K_rc)                                # reflect per RC
        
        K_rp = np.array(params['fiber']['K_rp'])
        z_rp = np.array(params['fiber']['z_rp'])
        
        K_rp = 10**(0.1*(K_rp))                                                    
        K_rp = np.ones(len(z_rp)) * np.sqrt(K_rp)                                #reflect per RP
    
        x = np.concatenate(([self.z_rc, K_rc],[z_rp, K_rp]), axis=1)    #coordinates and correspodning coefficients
        self.Z_K = x[:,x[0,:].argsort()]  
        
        """set_receiver"""
        Ladc = params['receiver']['Ladc']
        long_time = params['receiver']['long_time']
        self.Ndots = int(L / Ladc) 
        time_back = 2 * L * self.n0 / self.c  # duration of backscattering signal, s
        fast_time = np.linspace(0, time_back, self.Ndots) # time-window of backscattered signal, s
        self.Distance = fast_time * self.c / self.n0 / 2 # spatial-window of backscattered signal, s
        
        """set_pulse"""
        self.w_drift = params['pulse']['nyu_drift']
        self.lorentz_linewidth = params['pulse']['lorentz_linewidth']
        tau = params['pulse']['tau']
        fp = params['pulse']['fp']
        DF = params['pulse']['chirp_value']  # chirp value at pulse duration, Hz
        dw = params['pulse']['dw']  #  pulse modulation, Hz

        self.Time = np.arange(0, long_time, 1/fp)
        self.Timp = fast_time[fast_time <= tau]  # time-window of probe pulse, s
        H = np.heaviside(tau/2 - np.abs(self.Timp-tau/2),1) # probe pulse shape
        
        chirp = self.Timp / tau * DF
        wa = 2*np.pi* (dw + chirp/2)    # positive chirp, rad/s
        wb = 2*np.pi* (-(dw + DF) + chirp/2 )  # negative chirp, rad/s
        
        self.Pulse = H * np.exp(1j * wa * self.Timp) # chirped pulse
        self.Pulse_back_conj = np.conj(H * np.exp(1j * wb * self.Timp)) # for deconvolution

        self.NPulse=len(self.Timp)
        self.ZN = np.array(self.Z_K[0] / L * (self.Ndots - self.NPulse), dtype=np.int) #z-coordinate index in spatial-window "Distance" OTDR-trace
        
        """set_PZT"""
        self.K_mod = params['PZT']['K_mod']
        self.F_mod = params['PZT']['F_mod']
        self.Z_pzt = params['PZT']['Z_pzt']
        self.len_pzt = params['PZT']['len_pzt']
        
    
    def PZT(self, t):
        
        return self.K_mod * np.sin(2 * np.pi * self.F_mod * t)
       
        
    def get_amp(self, idx, t):
        
        w = self.w0 + 2 * np.pi * self.w_drift * t
        kv = w * self.n0 / self.c
        phase_pzt = 0
        
        for z_k_n, Zn in zip(self.Z_K.T, self.ZN):
            
            z_n, k_n = z_k_n
            
            if z_n >= self.Z_pzt and z_n < self.Z_pzt + self.len_pzt:
                
                phase_pzt = (z_n - self.Z_pzt) / self.len_pzt *  self.PZT(t)
               
            
            pulse_back = k_n * self.Pulse * np.exp(1j * self.phase_noise[idx]) * np.exp(2j * (kv * z_n + phase_pzt)) 
            
            try:                
                
                self.W[idx, Zn:Zn+self.NPulse] += pulse_back[:]
            
            except:
                
                pass

    
    @property
    def calc_phase_noise(self,):
        sigma_phase = np.sqrt(np.pi*2*(self.Timp[1] - self.Timp[0]) * self.lorentz_linewidth)
        np.random.seed(None) # unfixing random for phase_noise
        white_noise = np.random.randn(len(self.Time), self.NPulse)
        self.phase_noise = np.cumsum(sigma_phase * white_noise, axis=1)
        

    def calc_waterfall(self, params):
        """Complex amplitude of backscattering signal"""
        
        self.set_params(params)
        self.calc_phase_noise
        self.W = np.zeros((len(self.Time), self.Ndots), dtype='complex64')
        
        for idx, t in enumerate(self.Time):
            
            self.get_amp(idx, t)
        
        
        return self.W, self.Time, self.Distance, self.Pulse_back_conj, params



def calc_ASE(shape, G = 17, Be=10e6):
    NF = 7         #dB noise factor
    F = 10**(NF/10)
     #dB Gain
    G = 10**(G/10)
    h = 6.626*10e-34
    c = 3e8
    lamda = 1550e-9
    nu = c/lamda
    rho_ase = (F*G - 1)*h*nu
    P_ase = rho_ase * Be
    ASE_amp = (P_ase/2)**0.5
    size = (2, shape[0], shape[1])
    ASE =  ASE_amp * np.random.normal(0,1,size)
    I_ase, Q_ase = ASE[0], ASE[1]
    E_ase = I_ase + 1j * Q_ase
    return E_ase


def add_ASE(unnormed_E, G=15, Be=10e6, LEVEL=10**(5)):
    """Adding ASE noise to complex amplitudes
    G - gain in dB, Be - electrical bandwidth, LEVEL - norming parametr depends of signal power"""
    delta = np.mean(np.abs(unnormed_E[0])**2)*LEVEL
    normed_E = unnormed_E / np.sqrt(delta)
    En = normed_E + calc_ASE(shape=np.shape(normed_E), G=G, Be=Be)
    return En * G
    



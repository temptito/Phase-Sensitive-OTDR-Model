import numpy as np


class otdr():
    
    def __init__(self):
        self.test_params = {
                                'fiber': {'L': 1000, 'n_rc': 3, 'z_rp': [], 'K_rp': [-40]},
                                'receiver': {'Ladc': 1, 'long_time': 1},
                                'pulse': {'tau': 200e-9, 'fp': 1e3, 'w_drift': 0.1e6, 'Lorentz linewidth': 1e3},
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
        pulse: tau - pulse duration in s, fp - probing frequency in Hz, w_drift - linear carrier frequency drift in Hz/s,
        Lorentz linewidth - in Hz;
        PZT: K_mod - coefficient in rad, F_mod - frequency modualtion in Hz, 
        Z_pzt - PZT position in m, len_pzt - PZT length.
        """
        
        """set_fiber"""
        L = params['fiber']['L']
        n_rc = params['fiber']['n_rc']
        np.random.seed(1) # fixing random RC distribution
        self.z_rc = np.sort(np.random.rand(int(n_rc*L)))*L # RC's coordinates along fiber

        n_rc_ns = n_rc * self.c / 1e9 / self.n0                                         #РЦ на нc
        K_rc_ns = -80                                                                 #рассивание на нс
        K_rc = 10**(0.1*K_rc_ns) / n_rc_ns
        K_rc = np.ones(len(self.z_rc)) * np.sqrt(K_rc)                                #Коэффициент рассеивания на 1 РЦ
        
        K_rp = np.array(params['fiber']['K_rp'])
        z_rp = np.array(params['fiber']['z_rp'])
#         if K_rp:
        K_rp = 10**(0.1*(K_rp))                                                    #ВЧБР = -58 #
        K_rp = np.ones(len(z_rp)) * np.sqrt(K_rp)                                #Коэффициент рассеивания на 1 РТ,
    
        x = np.concatenate(([self.z_rc, K_rc],[z_rp, K_rp]), axis=1)    #Набор точек рассеваний и их коэффициентов
        self.Z_K = x[:,x[0,:].argsort()]  
        
        """set_receiver"""
        Ladc = params['receiver']['Ladc']
        long_time = params['receiver']['long_time']
        self.Ndots = int(L / Ladc) 
        time_back = 2 * L * self.n0 / self.c  # duration of backscattering signal, s
        fast_time = np.linspace(0, time_back, self.Ndots) # time-window of backscattered signal, s
        self.Distance = fast_time * self.c / self.n0 / 2 # spatial-window of backscattered signal, s
        
        """set_pulse"""
        self.w_drift = params['pulse']['w_drift']
        tau = params['pulse']['tau']
        fp = params['pulse']['fp']
        self.Lorentz_linewidth = params['pulse']['Lorentz linewidth']
        self.Time = np.arange(0, long_time, 1/fp)
        self.Timp = fast_time[fast_time <= tau]  # time-window of probe pulse, s
        self.Pulse = np.heaviside(tau/2 - np.abs(self.Timp-tau/2),1) # probe pulse  
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
            
            if Zn < 0: 
                
                self.W[idx, 0:Zn+self.NPulse] += pulse_back[:Zn+self.NPulse]

            elif Zn >= 0: # in case when part pulse behind of spatial-window
                
                self.W[idx, Zn:Zn+self.NPulse] += pulse_back[:]

            else: # in case when part pulse beyound of spatial-window
                
                self.W[idx, Zn:self.Ndots] += pulse_back[:self.Ndots - Zn]
                
    
    @property
    def calc_phase_noise(self,):
        sigma_phase = np.sqrt(np.pi*2*(self.Timp[1] - self.Timp[0]) * self.Lorentz_linewidth)
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
        
        
        return self.W, self.Time, self.Distance, params

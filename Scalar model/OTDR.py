import numpy as np

class otdr():
    
    def __init__(self):
        self.standart_params = params = {
                                        'fiber': {'L': 1000, 'n_rc': 3},
                                        'receiver': {'Ladc': 1, 'long_time': 1},
                                        'pulse': {'tau': 200e-9, 'fp': 1e3, 'w_drift': 6e6},
                                        'PZT': {'K_mod': 0, 'F_mod': 0, 'Z_pzt': 500, 'len_pzt': 30}
                                    }
        
        self.alpha = 0.0410517 * 1e-3 # attenuation, 1/m
        self.c = 300000000 # light speed, m/s
        self.n0 = 1.5      # refractive index
        self.lamda = 1550e-9 # wavelength, m
        self.kv = 2*np.pi/lamda # wavenumber, 1/m  
        self.w0 = kv * c / n0 # carrier frequency, rad/s
    
    def set_params(self, params):
        """
        setting parameters of 
        fiber: L - fiber length in m, n_rc - number of RC per meter in 1/m;
        receiver: Ladc - sampling frequency of receiver in m, T - time of capturing data in s;
        pulse: tau - pulse duration in s, fp - probing frequency in Hz, w_drift - linear carrier frequency drift in Hz/s;
        PZT: K_mod - coefficient in rad, F_mod - frequency modualtion in Hz, Z_pzt - PZT position in m, len_pzt - PZT length.
        """
        
        """set_fiber"""
        L = params['fiber']['L']
        n_rc = params['fiber']['n_rc']
        np.random.seed(1) # fixing random RC distribution
        self.z_rc = np.sort(np.random.rand(int(n_rc*L)))*L # RC's coordinates along fiber
        
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
        self.Time = np.arange(0, long_time, 1/fp)

        self.Timp = fast_time[fast_time <= tau]  # time-window of probe pulse, s
        self.Pulse = np.heaviside(tau/2 - np.abs(self.Timp-tau/2),1) # probe pulse  
        self.NPulse=len(self.Timp)
        
        self.ZN = np.array(self.z_rc / L * (self.Ndots - self.NPulse), dtype=np.int) # it z-coordinate index in spatial-window "Distance" OTDR-trace
        
        """set_PZT"""
        self.K_mod = params['PZT']['K_mod']
        self.F_mod = params['PZT']['F_mod']
        self.Z_pzt = params['PZT']['Z_pzt']
        self.len_pzt = params['PZT']['len_pzt']
        
    
    def PZT(self, t):
        
        return self.K_mod * np.sin(2 * np.pi * self.F_mod * t)
        
    def get_amp(self, t):
        
        w = self.w0 + 2 * np.pi * self.w_drift * t
        kv = w * n0 / c
        A = np.zeros(self.Ndots, dtype='complex64') #A(z) backscattering signal (complex amplitude) at reciver (z=0, t=n*2*z_k/c)
        phase_pzt = 0
        
        for z_n, Zn in zip(self.z_rc, self.ZN):
            
            if z_n >= self.Z_pzt and z_n < self.Z_pzt + self.len_pzt:
                
                phase_pzt = (z_n - self.Z_pzt) / self.len_pzt *  self.PZT(t)
                
                
            pulse_back = self.Pulse * np.exp(2j * (kv * z_n + phase_pzt)) # backscattered pulse at receiver (z=0)
            
            if Zn < 0: 
                
                A[0:Zn+self.NPulse] += pulse_back[:Zn+self.NPulse]

            elif Zn >= 0: # in case when part pulse behind of spatial-window
                
                A[Zn:Zn+self.NPulse] += pulse_back[:]

            else: # in case when part pulse beyound of spatial-window
                
                A[Zn:self.Ndots] += pulse_back[:self.Ndots - Zn]
                
                
        return A
    
    def get_waterfall(self, params):
        
        self.set_params(params)
        W = [self.get_amp(t) for t in self.Time]
        
        return W, self.Time, self.Distance, params


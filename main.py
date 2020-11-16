#Ali Keramatipour - 810196616
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class lloyds:
    def __init__(self, SIGMA, BITS, MEAN = 0, PRECISION = 0.005, SAMPLE_CNT = 10000):
        self.PRECISION = PRECISION
        self.MEAN = MEAN
        self.SIGMA = SIGMA
        self.Q = (1<<BITS)
        self.samples = np.random.normal(MEAN, SIGMA, SAMPLE_CNT)
        self.samples.sort()
        self.borders = []
        self.candidates = []
        parts = int(SAMPLE_CNT/self.Q)
        for i in range(1,self.Q):
            self.borders.append((self.samples[i * parts] + self.samples[i * parts - 1]) / 2)
        
    def update_candidates_borders(self):
        ranges = [[] for i in range(self.Q)]
        cnt = 0
        for s in self.samples:
            while cnt < self.Q - 1 and s > self.borders[cnt]:
                cnt = cnt + 1
            ranges[cnt].append(s)
        self.candidates.clear()
        
        for eachRange in ranges:
            sum_arr = []
            tmp = [float(val ** 2) for val in eachRange]
            p1_sum = sum(eachRange)
            p2_sum = sum(tmp)
            for i in range(len(eachRange)):
                dif = eachRange[i]
                tmp_sum = p2_sum - (2 * p1_sum * dif) + (len(eachRange) * (dif**2))
                sum_arr.append(tmp_sum)
            self.candidates.append(eachRange[np.argmin(sum_arr)])
        
        tmpBorders = self.borders.copy()
        self.borders.clear()
        for i in range(1,self.Q):
            self.borders.append((self.candidates[i] + self.candidates[i - 1]) / 2)
        for i in range(self.Q - 1):
            if abs(self.borders[i] - tmpBorders[i]) > self.PRECISION:
                return False
        
        return True
    
    def run(self):
        cycles = 0
        while not self.update_candidates_borders():
            cycles += 1
        print("cycles:", cycles)
    
    def draw_plots(self):
        x = np.linspace(self.MEAN - 3 * self.SIGMA, self.MEAN + 3 * self.SIGMA, 1000)
        
        plt.figure(figsize=(10, 5), num = "LLOYDS")
        plt.plot(x, stats.norm.pdf(x, self.MEAN, self.SIGMA))

        plot_height = stats.norm.pdf(0, self.MEAN, self.SIGMA)

        for x in self.borders:
            plt.plot([x, x],[0, plot_height], 'g--')
        
        for x in self.candidates:
            plt.plot([x, x],[0, plot_height], 'r-')
        
        plt.legend()
        plt.show()
    
    def print_outputs(self):
        print("Borders")
        for border in self.borders:
            print (border, ", ", end='', sep='')
        print()
        print("Candidates")
        for candidate in self.candidates:
            print (candidate, ", ", end='', sep='')
        print()


inst = lloyds(1, 3)
inst.run()
inst.draw_plots()
inst.print_outputs()

